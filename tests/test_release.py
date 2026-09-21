"""Regression tests for settings, data flow, geometry and user-edited exports.

Neural stages are intercepted in GUI tests; no weights or source data are needed.
Run: python -m unittest discover -s tests -v
"""
from pathlib import Path
import ast
import io
import json
import os
import sys
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'app'), str(ROOT)]
os.environ['MPLBACKEND'] = 'Agg'
import cv2
import numpy as np
import pandas as pd
import streamlit as st
from utils import analysis_runner as ar
from utils import session_state as ss
from components import results_display as rd
from modules import preprocessing as pre
from modules import runtime_config as rc
from modules import pipeline_cache as cache
from modules.distribution_analysis import calculate_distribution_metrics
from modules.shape_analysis import crop_and_zoom_particle
from modules.sam2_utils import filter_overlapping_masks_by_centroid, filter_background_masks


def masks_for(centers, shape=(128,128)):
    masks=[]
    for c in centers:
        a=np.zeros(shape,np.uint8);cv2.circle(a,c,3,1,-1)
        masks.append({'segmentation':a.astype(bool),'area':int(a.sum()),'bbox':list(cv2.boundingRect(a))})
    return masks


class ReleaseTests(unittest.TestCase):
    def test_noise2sr_settings_roundtrip(self):
        cfg=rc.resolve_noise2sr_config()
        self.assertEqual((cfg['patch_size'],cfg['batch_size'],cfg['epoch'],cfg['M']),(128,12,1500,50))
        state={}
        with patch.object(st,'session_state',state):
            ss.initialize_session_state()
            ss.apply_all_config(json.loads((ROOT/'configs/publication_final.json').read_text()))
            exported=ss.get_all_config()
            exported['preprocessing']['noise2sr']['patch_size']=96
            ss.apply_all_config(exported)
            self.assertEqual(exported,ss.get_all_config())
            self.assertEqual(ss.get_all_config()['preprocessing']['noise2sr']['patch_size'],96)

    def test_cache_tracks_execution_settings_and_runtime(self):
        with patch.object(cache,'runtime_signature',return_value={'source':'first'}):
            a=cache._build_signature('test.png',{'noise2sr':{'patch_size':128}})[1]
            b=cache._build_signature('test.png',{'noise2sr':{'patch_size':256}})[1]
        with patch.object(cache,'runtime_signature',return_value={'source':'changed'}):
            c=cache._build_signature('test.png',{'noise2sr':{'patch_size':128}})[1]
        self.assertNotEqual(a,b);self.assertNotEqual(a,c)

    def test_runtime_digest_tracks_yaml_contents(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'configs').mkdir()
            config=root/'configs/test.yaml';config.write_text('patch_size: 128')
            with patch.object(rc,'ROOT',root):
                a=rc.runtime_signature()['source_sha256']
                config.write_text('patch_size: 256')
                b=rc.runtime_signature()['source_sha256']
            self.assertNotEqual(a,b)

    def test_gui_clip_receives_processed_image_and_effective_config(self):
        state={};seen=[];seen_noise=[]
        masks=masks_for([(50,50)])
        bounds=pre.extract_boundary_pixels_dict(masks)
        def noise(image,config,verbose=True):
            seen_noise.append(config)
            return np.full_like(image,203),{'config':dict(config)}
        def shape(image,*args):
            seen.append(float(image.mean()))
            return {'shapes':['Circle'],'confidences':[.9],'shape_counts':{'Circle':1},'color_map':{}}
        with tempfile.TemporaryDirectory() as td, patch.object(st,'session_state',state), \
             patch.object(pre,'apply_bm3d_denoising',lambda image,**kw:image.copy()), \
             patch.object(pre,'check_bm3d_available',return_value=True), \
             patch.object(pre,'check_noise2sr_available',return_value=True), \
             patch.object(pre,'apply_noise2sr_network',side_effect=noise), \
             patch.object(ar,'run_sam_segmentation',return_value=(masks,[(50,50)],bounds)), \
             patch.object(ar,'run_shape_analysis',side_effect=shape), \
             patch.object(ar,'create_shape_overlay',lambda *a,**k:__import__('matplotlib.pyplot',fromlist=['']).subplots()[0]), \
             patch.object(ar,'create_dashboard_image',return_value=np.zeros((2,2,3),np.uint8)), \
             patch.object(ar,'CACHE_ROOT_DIR',td):
            ss.initialize_session_state()
            state['analysis_config'].update(enable_size=False,enable_distribution=False,enable_spatial_uniformity=False)
            result=ar.run_analysis(np.full((128,128,3),17,np.uint8),'test.png')
            self.assertIsNone(result.get('error'),result.get('error'))
            self.assertEqual(seen,[203.0])
            self.assertEqual(seen_noise[0]['batch_size'],12)
            self.assertEqual(seen_noise[0]['patch_size'],128)
            state['analysis_results']={'test.png':result}
            with ZipFile(io.BytesIO(rd.create_results_zip())) as z:
                self.assertIsNone(z.testzip())
                self.assertTrue(any(x.endswith('.xlsx') for x in z.namelist()))
                self.assertTrue(any(x.endswith('.png') for x in z.namelist()))

    def test_pf_unavailable_for_zero_one_or_collinear_interior(self):
        examples=[[(20,20),(100,20),(60,100)],
                  [(20,20),(100,20),(100,100),(20,100),(60,60)],
                  [(20,60),(60,60),(100,60)]]
        for centers in examples:
            with self.subTest(centers=centers):
                masks=masks_for(centers)
                d=ar.run_distribution_analysis(centers,pre.extract_boundary_pixels_dict(masks),(128,128),1,'px',{})
                self.assertEqual(d['status'],'unavailable')
                self.assertIsNone(d['sui']);self.assertTrue(d['reason'])
                self.assertLess(d['n_pf'],2)

    def test_pf_sample_sd_and_scale_invariance(self):
        self.assertAlmostEqual(calculate_distribution_metrics([1,2,3])['sui'],2/3)
        with self.assertRaises(ValueError):calculate_distribution_metrics([1])
        centers=[(20,20),(100,20),(100,100),(20,100),(45,60),(75,60)]
        bounds=pre.extract_boundary_pixels_dict(masks_for(centers))
        d=ar.run_distribution_analysis(centers,bounds,(128,128),1,'px',{})
        scaled=ar.run_distribution_analysis(centers,bounds,(128,128),.5,'nm',{})
        self.assertEqual(d['status'],'complete');self.assertEqual(d['n_pf'],2)
        self.assertAlmostEqual(d['sui'],scaled['sui'])
        np.testing.assert_allclose(np.asarray(d['voronoi_areas_real'])*.25,scaled['voronoi_areas_real'])

    def test_deletion_rebuilds_empty_exports_and_cumulative_provenance(self):
        centers=[(20,20),(100,20),(60,100)];masks=masks_for(centers)
        bounds=pre.extract_boundary_pixels_dict(masks)
        image=np.zeros((128,128,3),np.uint8)
        state={}
        with patch.object(st,'session_state',state):
            ss.initialize_session_state()
            old={'particle_count':3,'filtered_masks':masks,'centroids':centers,
                 'size':ar.run_size_analysis(masks,bounds,1,'px',{}),
                 'distribution':{'sui':.987,'voronoi_areas_real':[1,2,3]},
                 'shape':{'shapes':['Circle','Triangle','Circle'],'confidences':[.9,.8,.7]},
                 'dashboard_image':np.full((2,2,3),77,np.uint8)}
            rd.recalculate_after_deletion('test.png',image,old,{1,2},masks,centers)
            one=state['analysis_results']['test.png']
            self.assertEqual(len(rd.create_particle_dataframe(one)),1)
            self.assertEqual(one['shape']['shapes'],['Circle'])
            self.assertIsNone(one['distribution']['sui'])
            self.assertFalse(np.array_equal(one['dashboard_image'],old['dashboard_image']))
            rd.recalculate_after_deletion('test.png',image,one,{0},one['filtered_masks'],one['centroids'])
            empty=state['analysis_results']['test.png']
            self.assertEqual(empty['size']['count'],0)
            self.assertEqual(empty['shape']['shape_counts'],{})
            self.assertEqual(len(pd.read_csv(io.StringIO(rd.create_particle_csv(empty)))),0)
            self.assertEqual(empty['execution_provenance']['manual_postprocessing']['deleted_particle_ids'],[1,2,3])
            with pd.ExcelFile(io.BytesIO(rd.create_excel_report(empty))) as book:
                self.assertTrue(book.sheet_names)

    def test_crop_includes_last_pixel_and_uses_mask_not_intensity(self):
        mask=np.zeros((20,20),bool);mask[5,5]=True
        image=np.zeros((20,20,3),np.uint8);image[5,5]=173
        out=np.asarray(crop_and_zoom_particle(image,segmentation_mask=mask,padding_ratio=0))
        self.assertTrue(np.all(out==173))
        image[:]=0
        self.assertTrue(np.all(np.asarray(crop_and_zoom_particle(image,segmentation_mask=mask))==0))

    def test_nested_removes_larger_mask(self):
        outer=np.zeros((100,100),bool);outer[20:60,20:60]=True
        inner=np.zeros_like(outer);inner[30:40,30:40]=True
        out=filter_overlapping_masks_by_centroid([{'segmentation':outer},{'segmentation':inner}])
        self.assertEqual([int(m['segmentation'].sum()) for m in out],[100])

    def test_background_uses_bbox_area_not_edge_length(self):
        records=[{'bbox':[0,0,90,100],'id':'exact_90_percent'},
                 {'bbox':[0,0,91,100],'id':'above_90_percent'},
                 {'bbox':[0,30,100,10],'id':'thin_full_width'},
                 {'bbox':[5,5,90,90],'id':'within_5_of_all_edges'}]
        kept=filter_background_masks(records,(100,100))
        self.assertEqual([m['id'] for m in kept],['exact_90_percent','thin_full_width'])

    def test_preprocessing_cache_rejects_missing_or_changed_metadata(self):
        with tempfile.TemporaryDirectory() as td, patch.object(rc,'runtime_signature',return_value={'source':'current'}):
            path=Path(td)/'processed.png'
            self.assertFalse(rc.preprocessing_record_matches(path,(128,128)))
            record=rc.preprocessing_record((128,128))
            path.with_suffix('.json').write_text(json.dumps({'release_preprocessing':record}))
            self.assertTrue(rc.preprocessing_record_matches(path,(128,128)))
            record['noise2sr']['batch_size']=8
            path.with_suffix('.json').write_text(json.dumps({'release_preprocessing':record}))
            self.assertFalse(rc.preprocessing_record_matches(path,(128,128)))

    def test_deletion_regenerates_available_spatial_images(self):
        centers=[(20,20),(100,20),(100,100),(20,100),(45,60),(75,60),(60,80)]
        masks=masks_for(centers)
        state={}
        with patch.object(st,'session_state',state):
            ss.initialize_session_state();state['analysis_config']['enable_shape']=False
            old={'particle_count':7,'filtered_masks':masks,'centroids':centers}
            rd.recalculate_after_deletion('test.png',np.zeros((128,128,3),np.uint8),old,{6},masks,centers)
            dist=state['analysis_results']['test.png']['distribution']
            self.assertEqual(dist['status'],'complete')
            self.assertIn('spatial_distribution_image',dist)
            self.assertIn('voronoi_image',dist)

    def test_external_pf_unavailable_has_no_stale_plot(self):
        from types import SimpleNamespace
        sys.path.insert(0,str(ROOT/'analysis'))
        import run_case_study_batch as case
        centers=[(20,20),(100,20),(100,100),(20,100),(60,60)]
        with tempfile.TemporaryDirectory() as td:
            output=Path(td);args=SimpleNamespace(output_dir=output)
            spec=case.InputSpec('gt_evaluation',output/'006.tif')
            destination=case.image_dir(output,spec);destination.mkdir(parents=True)
            stale=destination/'spatial_distribution.png';stale.write_bytes(b'old')
            with patch.object(case,'load_masks',return_value=masks_for(centers)), \
                 patch.object(case,'read_bgr_uint8',return_value=np.zeros((128,128,3),np.uint8)), \
                 patch.object(case,'spatial_source_fingerprint',return_value={'source':'test'}):
                result=case.run_spatial_for_image(spec,args,output/'processed.png','testhash')
            self.assertEqual(result['status'],'unavailable')
            self.assertEqual(result['spatial_metrics']['n_pf'],1)
            self.assertIsNone(result['spatial_metrics']['spatial_uniformity_index'])
            self.assertFalse(stale.exists())

    def test_stage_crossing_uses_q_representative_not_any_candidate(self):
        names={'total_noise_valid_ratio','_combo_quality_key','_select_from_stage','find_optimal_stage','calculate_growth_rate'}
        tree=ast.parse((ROOT/'analysis/sam_param_optimizer.py').read_text(encoding='utf-8-sig'))
        ns={'np':np,'CONSERVATIVE_GROWTH_THRESHOLD':.5}
        exec(compile(ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[]),'screening','exec'),ns)
        stages=[[{'valid_masks':10,'noise_masks':0}],
                [{'valid_masks':20,'noise_masks':2},{'valid_masks':11,'noise_masks':1}],
                [{'valid_masks':22,'noise_masks':3}]]
        # Stage 2 second candidate has R=1, but its q is smaller: it is the representative.
        self.assertEqual(ns['find_optimal_stage'](stages)[:2],(0,0))
        stages[1][1]={'valid_masks':11,'noise_masks':5} # larger q: ignored for crossing
        # Representative R=.2 then equality R=.5: stop and return stage 2, candidate 1.
        self.assertEqual(ns['find_optimal_stage'](stages)[:2],(1,0))


if __name__=='__main__':
    unittest.main()
