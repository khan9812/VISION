"""Independently check released numerical artifacts against their included rows."""
import ast
from pathlib import Path
import json
import warnings
warnings.filterwarnings('ignore',category=UserWarning)
import numpy as np
import pandas as pd
from scipy import stats

import argparse
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=Path('workspace_outputs/reproduction'))
args=parser.parse_args()
OUT=args.output_dir.resolve()
OUT.mkdir(parents=True,exist_ok=True)
REPO=Path(__file__).resolve().parents[1]
RESULTS=REPO/'results' 
final=json.loads((RESULTS/'final_results.json').read_text())
report={'checks':[]}

def check(name, observed, expected, atol=5e-7):
    a=np.asarray(observed,dtype=float)
    b=np.asarray(expected,dtype=float)
    report['checks'].append({'name':name,'observed':a.tolist(),'expected':b.tolist(),'pass':bool(np.allclose(a,b,atol=atol,rtol=0,equal_nan=True))})

screen=pd.read_excel(RESULTS/'sam_screening/optimization_results.xlsx',sheet_name='All Images')
pairs=pd.read_excel(RESULTS/'sam_screening/optimization_results.xlsx',sheet_name='Pair Metrics')
names={'total_noise_valid_ratio','_combo_quality_key','_select_from_stage','find_optimal_stage','calculate_growth_rate'}
tree=ast.parse((REPO/'analysis/sam_param_optimizer.py').read_text(encoding='utf-8-sig'))
subset=ast.Module(body=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names],type_ignores=[])
ns={'CONSERVATIVE_GROWTH_THRESHOLD':0.5,'np':np}
exec(compile(subset,'released_screening_functions','exec'),ns)
differences=[]
for image,g in pairs.groupby('Image',sort=False):
    stages=[]
    for stage,sg in g.groupby('Stage',sort=True):
        stages.append([{'stage':int(stage)-1,'combo_idx':int(r.Combo)-1,'valid_masks':int(r.TP),'noise_masks':int(r.FP),'pred_iou_thresh':r.pred_iou_thresh,'stability_score_thresh':r.stability_score_thresh} for r in sg.sort_values('Combo').itertuples()])
    si,ci,_=ns['find_optimal_stage'](stages)
    selected=stages[si][ci]
    saved=screen.loc[screen.Image==image].iloc[0]
    if (selected['pred_iou_thresh'],selected['stability_score_thresh'])!=(saved.pred_iou_thresh,saved.stability_score_thresh):
        differences.append(image)
check('screening_image_count',len(screen),256,0)
check('screening_pair_count',len(pairs),256*19,0)
check('screening_selection_mismatches',len(differences),0,0)
check('screening_mode_count',((screen.pred_iou_thresh==.95)&(screen.stability_score_thresh==.8)).sum(),140,0)

size=RESULTS/'size_validation/size_validation_metrics.xlsx'
d=pd.read_excel(size,sheet_name='Matched_Particles')
c=pd.read_excel(size,sheet_name='Per_Image_Coverage').sort_values('Image_ID')
ids=c.Image_ID.tolist()
agg=d.groupby('Image_ID').agg(n=('IoU','size'),iou=('IoU','sum'),error=('Absolute_Size_Error_px2','sum')).reindex(ids,fill_value=0)
indices=np.random.default_rng(42).integers(0,len(ids),size=(10000,len(ids)))
denom=agg.n.to_numpy()[indices].sum(axis=1)
boot_iou=agg.iou.to_numpy()[indices].sum(axis=1)/denom
boot_mae=agg.error.to_numpy()[indices].sum(axis=1)/denom
size_final=final['size_validation']
for name,observed,expected in [
    ('size_image_count',len(c),size_final['images']),('size_gt_count',c.GT_Particle_Count.sum(),size_final['gt_particles']),
    ('size_prediction_count',c.Predicted_Particle_Count.sum(),size_final['predictions']),('size_matched_count',len(d),size_final['matched_particles']),
    ('size_pooled_iou',d.IoU.mean(),size_final['mean_matched_iou']),('size_pooled_mae',d.Absolute_Size_Error_px2.mean(),size_final['area_mae_px2']),
    ('size_pooled_precision',c.Matched_Particle_Count.sum()/c.Predicted_Particle_Count.sum(),size_final['pooled_precision']),
    ('size_macro_recall',c.Recall.mean(),size_final['image_macro_recall']),
    ('size_iou_ci',np.percentile(boot_iou,[2.5,97.5]),size_final['mean_matched_iou_ci95']),
    ('size_mae_ci',np.percentile(boot_mae,[2.5,97.5]),size_final['area_mae_px2_ci95'])]:check(name,observed,expected)
report['size_bootstrap']={'unit':'image','images':len(ids),'resamples':10000,'seed':42,'zero_match_images':int((agg.n==0).sum()),'denominator_range':[int(denom.min()),int(denom.max())]}

sh=pd.read_excel(RESULTS/'shape_validation/shape_validation_results.xlsx',sheet_name='Detailed Results')
labels=['Circle','Triangle','Quadrilateral','Hexagon','Irregular']
image_ids=sorted(sh.image_name.unique())
image_index={v:i for i,v in enumerate(image_ids)}
label_index={v:i for i,v in enumerate(labels)}
cm=np.zeros((len(image_ids),5,5),dtype=np.int64)
for row in sh.itertuples():cm[image_index[row.image_name],label_index[row.gt_shape],label_index[row.pred_shape]]+=1
def metrics(cm):
    tp=np.diagonal(cm,axis1=-2,axis2=-1)
    den=cm.sum(axis=-1)+cm.sum(axis=-2)
    f1=np.divide(2*tp,den,out=np.zeros_like(tp,dtype=float),where=den>0).mean(axis=-1)
    accuracy=tp.sum(axis=-1)/cm.sum(axis=(-1,-2))
    return accuracy,f1
acc,f1=metrics(cm.sum(axis=0))
bs=np.random.default_rng(42).integers(0,len(cm),size=(10000,len(cm)))
bacc,bf1=metrics(cm[bs].sum(axis=1))
check('shape_particles',len(sh),403,0)
check('shape_images',len(image_ids),100,0)
check('shape_accuracy',acc,final['shape_validation']['accuracy'])
check('shape_macro_f1',f1,final['shape_validation']['macro_f1'])
check('shape_accuracy_ci',np.percentile(bacc,[2.5,97.5]),final['shape_validation']['accuracy_ci95'])
check('shape_macro_f1_ci',np.percentile(bf1,[2.5,97.5]),final['shape_validation']['macro_f1_ci95'])
report['shape_bootstrap_source_in_package']=True

pf=pd.read_excel(RESULTS/'pf_sui/pf_sui_validation_results.xlsx',sheet_name='All_Results')
pfsummary=pd.read_excel(RESULTS/'pf_sui/pf_sui_validation_results.xlsx',sheet_name='Condition_Summary_CI').set_index('Case')
check('pf_observations',len(pf),540,0)
for index,label in enumerate('ABCDEF'):
    vals=pf.loc[pf.case==label,'di'].to_numpy()
    check('pf_mean_'+label,vals.mean(),final['pf_sui']['means'][label])
    bs=np.random.default_rng(42+index).integers(0,len(vals),size=(10000,len(vals)))
    check('pf_ci_'+label,np.percentile(vals[bs].mean(axis=1),[2.5,97.5]),pfsummary.loc[label,['SUI_CI95_Low','SUI_CI95_High']].to_numpy())
paired=pf.pivot(index=['version','repeat'],columns='case',values='num_voronoi_cells')
report['pf_pair_eligible_count_differences']={a+'-'+b:int((paired[a]!=paired[b]).sum()) for a,b in [('A','B'),('C','D'),('E','F')]}

pre=pd.read_excel(RESULTS/'preprocessing/preprocessing_metrics.xlsx',sheet_name='Method Summary')
record=pre.loc[pre.Method_Key=='2_bm3d_noise2sr'].iloc[0]
check('preprocessing_image_count',record.N,360,0)
check('preprocessing_macro_f1',record.Mean_F1,final['preprocessing']['macro_f1'])
f1raw=pd.read_csv(RESULTS/'preprocessing/paired_f1_360_images.csv')
for metric,col,estimate,ci in [('f1','BM3D_Noise2SR_F1',final['preprocessing']['macro_f1'],final['preprocessing']['macro_f1_ci95']),('delta_f1','delta_F1',final['preprocessing']['delta_f1_vs_bm3d'],final['preprocessing']['delta_f1_vs_bm3d_ci95'])]:
    vals=f1raw[col].to_numpy();delta=stats.t.ppf(.975,len(vals)-1)*stats.sem(vals)
    check('preprocessing_raw_'+metric,vals.mean(),estimate)
    check('preprocessing_raw_'+metric+'_ci',[vals.mean()-delta,vals.mean()+delta],ci)
report['preprocessing_independent_source']='Included per-image_metrics.csv and paired_f1_360_images.csv; frozen model predictions.'
raw=pd.read_csv(RESULTS/'preprocessing/per_image_metrics.csv')
check('preprocessing_four_method_records',len(raw),1440,0)
check('preprocessing_f1_from_counts',raw['F1 Score'],2*raw.TP/(2*raw.TP+raw.FP+raw.FN),1e-12)
for method,group in raw.groupby('Method Key'):
    row=pre.loc[pre.Method_Key==method].iloc[0]
    check('preprocessing_mean_'+method,group['F1 Score'].mean(),row.Mean_F1)
report['f1_exactly_one_by_method']=raw.groupby('Method Key')['F1 Score'].agg(lambda x:int((x==1).sum())).to_dict()
report['bootstrap_estimands']={
 'size':'Image-cluster resampling of 200 images; pooled sum IoU or absolute error divided by resampled matched-pair count.',
 'shape':'Image-cluster resampling of 100 images; summed confusion matrix; pooled accuracy and equal-weight mean F1 over five fixed labels; absent-class F1=0.',
 'pf':'Within-condition mean; 90 realization records resampled per condition, seed 42 + condition index.',
 'preprocessing':'Image macro means and paired differences; Student-t intervals, not bootstrap.'}
external=final['external_case_study']
case=RESULTS/'external_case_study'
spatial=pd.read_csv(case/'gt_spatial_voronoi_areas.csv',dtype={'image_id':str})
shape_summary=pd.read_csv(case/'gt_shape_summary.csv',dtype={'image_id':str}).set_index('image_id').loc[external['image_ids']]
check('external_particle_counts',shape_summary.particle_count,external['particle_counts'],0)
check('external_individual_counts',shape_summary['individual particle'],external['individual_counts'],0)
check('external_cluster_counts',shape_summary['particle aggregate'],external['cluster_counts'],0)
for i,image_id in enumerate(external['image_ids']):
    vals=spatial.loc[spatial.image_id==image_id,'voronoi_area_px2'].to_numpy()
    check('external_n_pf_'+image_id,len(vals),external['pf_eligible_counts'][i],0)
    check('external_pf_sui_'+image_id,1/(1+vals.std(ddof=1)/vals.mean()),external['pf_sui'][i],1e-12)
report['summary']={'checks':len(report['checks']),'passed':sum(c['pass'] for c in report['checks']),'failed':[c['name'] for c in report['checks'] if not c['pass']]}
(OUT/'numeric_verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report['summary'],indent=2))
print('PF paired eligible differences',report['pf_pair_eligible_count_differences'])

raise SystemExit(0 if not report['summary']['failed'] else 1)
