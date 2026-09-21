"""
Visualization Module for Integrated Particle Analysis Dashboard

This module provides the integrated 2x3 dashboard visualization that combines
size, spatial uniformity, and shape analysis results into a single comprehensive view.

Layout:
    Row 1: Size Histogram | Size Overlay (heatmap + labels)
    Row 2: Spatial Uniformity Violin Plot | Voronoi Overlay (heatmap + labels)
    Row 3: Shape Pie Chart | Shape Overlay (colored masks)
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, Optional, Tuple
import matplotlib.patches as mpatches

# Core scientific computing libraries
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.patches import Circle
import random
import os

# Scipy and scientific computing
from scipy.spatial import Voronoi, voronoi_plot_2d, distance_matrix
from scipy.stats import iqr
from scipy.integrate import simps

# Machine learning and clustering
from sklearn.cluster import DBSCAN
try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False

# Geometry and shape analysis
from shapely.geometry import MultiPoint, Polygon, Point, MultiPolygon
from shapely.ops import unary_union

# SAM imports
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator

# Utility libraries
from itertools import combinations
from math import sqrt
from collections import Counter
from modules.color_palette import (
    PALETTE_NEUTRAL,
    make_value_colormap,
    palette_bgr255,
    palette_hex,
    palette_rgb,
    palette_sequence,
)
from modules.scientific_plotting import (
    plot_shape_composition,
    plot_size_distribution,
    plot_spatial_distribution,
)

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    colors = palette_sequence(len(sorted_anns))
    for idx, ann in enumerate(sorted_anns):
        m = ann['segmentation']
        r, g, b = colors[idx % len(colors)]
        color_mask = np.array([r, g, b, 0.35])
        img[m] = color_mask
    ax.imshow(img)

def nothing(x):
    pass


def visualize_and_get_centroids(image, filtered_masks):
    particle_part = image.copy()
    centroids = []

    for idx, mask_info in enumerate(filtered_masks):
        mask = mask_info['segmentation']
        binary_mask = (mask > 0).astype(np.uint8) * 255

        # 컨투어 탐지
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = len(contours)

        centroid_found = False
        
        if num_contours > 0:
            # Zero area가 아닌 컨투어만 남기고 정렬 (가장 작은 순으로)
            valid_contours = [contour for contour in contours if cv2.contourArea(contour) > 0]
            
            if valid_contours:
                # 가장 작은 컨투어를 선택
                smallest_contour = min(valid_contours, key=cv2.contourArea)

                # 중심점 계산
                M = cv2.moments(smallest_contour)
                if M["m00"] != 0:
                    centroid_x = int(M["m10"] / M["m00"])
                    centroid_y = int(M["m01"] / M["m00"])
                    centroids.append((centroid_x, centroid_y))
                    cv2.circle(particle_part, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
                    centroid_found = True
        
        # Fallback: contour 추출 실패 시 bbox 중심점 사용
        if not centroid_found:
            bbox = mask_info.get('bbox', None)
            if bbox is not None:
                x, y, w, h = bbox
                centroid_x = int(x + w / 2)
                centroid_y = int(y + h / 2)
                centroids.append((centroid_x, centroid_y))
                cv2.circle(particle_part, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
                print(f"Mask {idx}: Using bbox center as fallback ({centroid_x}, {centroid_y})")
            else:
                # Last resort: mask의 True 픽셀들의 평균 위치
                ys, xs = np.where(mask > 0)
                if len(xs) > 0:
                    centroid_x = int(np.mean(xs))
                    centroid_y = int(np.mean(ys))
                    centroids.append((centroid_x, centroid_y))
                    cv2.circle(particle_part, (centroid_x, centroid_y), 5, (0, 255, 0), -1)
                    print(f"Mask {idx}: Using pixel mean as fallback ({centroid_x}, {centroid_y})")
                else:
                    # 이 경우는 정말 발생하면 안 됨 - mask가 완전히 비어있음
                    print(f"ERROR: Mask {idx} is completely empty! This should never happen.")
                    # 그래도 리스트 길이를 맞추기 위해 (0,0) 추가
                    centroids.append((0, 0))
    
    return particle_part, centroids

def filter_masks_based_on_centroid_containment(filtered_masks, centroids):
    masks_to_keep = set(range(len(filtered_masks)))  # 인덱스로 마스크 추적
    mask_centroid_map = {i: centroids[i] for i in range(len(centroids))}

    # 각 마스크와 그 centroid가 다른 마스크 내부에 있는지 체크
    for i, mask_dict in enumerate(filtered_masks):
        if i not in masks_to_keep:
            continue  # 이미 제거된 마스크는 건너뜀

        mask = mask_dict['segmentation']

        for j, centroid in mask_centroid_map.items():
            if j == i or j not in masks_to_keep:
                continue  # 자신의 centroid와 이미 제거된 centroid는 제외

            # centroid의 좌표가 마스크 내부에 있는지 확인
            x, y = int(centroid[0]), int(centroid[1])
            if (mask > 0).astype(np.uint8)[y, x] > 0:  # centroid가 마스크 내부에 있을 때
                # 마스크 크기 비교
                mask_to_compare = filtered_masks[j]['segmentation']
                if is_larger_mask(mask, mask_to_compare):
                    masks_to_keep.discard(i)  # 크기가 큰 마스크 (mask i) 삭제
                    break
                elif is_larger_mask(mask_to_compare, mask):
                    masks_to_keep.discard(j)  # 크기가 큰 마스크 (mask j) 삭제

    # 필터링된 마스크 반환
    filtered_masks = [filtered_masks[i] for i in masks_to_keep]
    centroids = [centroids[i] for i in masks_to_keep]

    return filtered_masks, centroids


def find_optimal_alpha(filtered_centroids):
    """Convex hull and strict 5-pixel inward centroid inclusion.

    The historical helper name is retained. An empty buffer retains no
    centroids; boundary particles are never substituted for interior ones.
    """
    if filtered_centroids is None or len(filtered_centroids) < 3:
        return [], None, [], 0.0
    hull = MultiPoint(np.asarray(filtered_centroids)).convex_hull
    if not isinstance(hull, Polygon) or hull.is_empty or not hull.is_valid:
        return [], None, [], 0.0
    interior = hull.buffer(-5.0)
    inside = list(dict.fromkeys(
        tuple(point) for point in filtered_centroids
        if not interior.is_empty and interior.contains(Point(point))
    ))
    return inside, [hull], list(hull.exterior.coords), 0.0

def calculate_polygon_area(vertices):
    n = len(vertices)
    area = 0.0
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0

def calculate_filtered_areas_within_concave_hull(vor, boundary_pixels, concave_vertices, inside_centroids):
    areas = []
    polygons = []
    
    # Concave Hull 다각형 생성
    concave_polygon = Polygon(concave_vertices)
    
    # boundary_pixels에서 filtered_centroids에 해당하는 key들만 선택
    selected_points = []
    excluded_keys = set(boundary_pixels.keys()) - set(inside_centroids)  # 제외된 key 확인용

    # filtered_centroids에 해당하는 key의 value들을 수집
    for key in inside_centroids:
        if key in boundary_pixels:
            selected_points.extend(boundary_pixels[key])
    
    selected_points = np.array(selected_points)  # 선택된 포인트들을 numpy 배열로 변환

    # Voronoi 셀 영역을 계산 (전체 Voronoi 다이어그램은 boundary_pixels의 모든 포인트들로 이미 생성됨)
    for region_index, point in enumerate(vor.points):
        region = vor.regions[vor.point_region[region_index]]
        if not -1 in region:  # 무한 영역은 무시
            polygon = Polygon([vor.vertices[i] for i in region])
            if polygon.is_valid and any(np.array_equal(point, sp) for sp in selected_points):
                # Voronoi 다각형이 Concave Hull과 교차하는지 확인
                intersection = polygon.intersection(concave_polygon)
                if intersection.is_empty or not isinstance(intersection, (Polygon, MultiPolygon)):
                    continue
                
                # 교차하는 부분의 넓이 계산
                areas.append(intersection.area)
                polygons.append(intersection)
    

    return areas, polygons, excluded_keys, selected_points

def calculate_total_area_per_key_pixels(vor, boundary_pixels, concave_vertices, inside_centroids):
    key_area_map = {}  # 각 key에 대한 통합된 넓이 저장
    key_polygons_map = {}  # 각 key에 대한 통합된 다각형 저장
    
    # Concave Hull 다각형 생성
    concave_polygon = Polygon(concave_vertices)
    
    # boundary_pixels에서 filtered_centroids에 해당하는 key들만 선택
    excluded_keys = set(boundary_pixels.keys()) - set(inside_centroids)  # 제외된 key 확인용
    
    # 각 key에 대해 Voronoi 다각형들을 통합
    for key, points in boundary_pixels.items():
        if key in inside_centroids:
            selected_polygons = []
            for region_index, point in enumerate(vor.points):
                if any(np.array_equal(point, np.array(p)) for p in points):
                    region = vor.regions[vor.point_region[region_index]]
                    if -1 in region:
                        continue
                    
                    polygon = Polygon([vor.vertices[i] for i in region])
                    if polygon.is_valid:
                        # Voronoi 다각형이 Concave Hull과 교차하는지 확인
                        intersection = polygon.intersection(concave_polygon)
                        if not intersection.is_empty and isinstance(intersection, (Polygon, MultiPolygon)):
                            selected_polygons.append(intersection)
            
            # key에 해당하는 다각형들을 통합
            if selected_polygons:
                unified_polygon = unary_union(selected_polygons)
                key_polygons_map[key] = unified_polygon
                key_area_map[key] = unified_polygon.area
    
    return key_area_map, key_polygons_map, excluded_keys

def calculate_areas_per_mask_pixels(boundary_pixels):
    # 각 마스크의 넓이를 저장할 딕셔너리
    mask_area_map = {}
    
    # boundary_pixels에 있는 각 키에 대해 개별 폴리곤 생성 및 면적 계산
    for key, points in boundary_pixels.items():
        polygon = Polygon(points)
        if polygon.is_valid:
            mask_area_map[key] = polygon.area  # 픽셀 단위 면적 저장
        else:
            print(f"Invalid polygon for key {key}")
    
    return mask_area_map


def postprocess_text(text):
    # Replace certain characters to correct recognition
    corrected_text = text.replace('@', '0')
    corrected_text = corrected_text.replace('s', '5')
    corrected_text = corrected_text.replace('S', '5')
    corrected_text = corrected_text.replace('o', '0')
    corrected_text = corrected_text.replace('O', '0')
    return corrected_text

def calculate_total_area_per_key_real(vor, boundary_pixels, concave_vertices, filtered_centroids, pixel_to_real_length):
    key_area_map = {}  # 각 key에 대한 통합된 실제 넓이 저장
    key_polygons_map = {}  # 각 key에 대한 통합된 다각형 저장
    
    # Concave Hull 다각형 생성
    concave_polygon = Polygon(concave_vertices)
    
    # boundary_pixels에서 filtered_centroids에 해당하는 key들만 선택
    excluded_keys = set(boundary_pixels.keys()) - set(filtered_centroids)  # 제외된 key 확인용
    
    # 각 key에 대해 Voronoi 다각형들을 통합
    for key, points in boundary_pixels.items():
        if key in filtered_centroids:
            selected_polygons = []
            for region_index, point in enumerate(vor.points):
                if any(np.array_equal(point, np.array(p)) for p in points):
                    region = vor.regions[vor.point_region[region_index]]
                    if -1 in region:
                        continue
                    
                    polygon = Polygon([vor.vertices[i] for i in region])
                    if polygon.is_valid:
                        # Voronoi 다각형이 Concave Hull과 교차하는지 확인
                        intersection = polygon.intersection(concave_polygon)
                        if not intersection.is_empty and isinstance(intersection, (Polygon, MultiPolygon)):
                            selected_polygons.append(intersection)
            
            # key에 해당하는 다각형들을 통합
            if selected_polygons:
                unified_polygon = unary_union(selected_polygons)
                key_polygons_map[key] = unified_polygon
                
                # 실제 넓이 계산
                pixel_area = unified_polygon.area
                real_area = pixel_area * (pixel_to_real_length ** 2)
                key_area_map[key] = real_area
    
    return key_area_map, key_polygons_map, excluded_keys

def calculate_areas_per_mask_real(boundary_pixels, pixel_to_real_length):
    # 각 마스크의 실제 넓이를 저장할 딕셔너리
    mask_area_map = {}
    
    # boundary_pixels에 있는 각 키에 대해 개별 폴리곤 생성 및 실제 면적 계산
    for key, points in boundary_pixels.items():
        polygon = Polygon(points)
        if polygon.is_valid:
            pixel_area = polygon.area  # 픽셀 단위 면적
            real_area = pixel_area * (pixel_to_real_length ** 2)  # 실제 면적으로 변환
            mask_area_map[key] = real_area
        else:
            print(f"Invalid polygon for key {key}")
    
    return mask_area_map

def calculate_distance(point1, point2):
    return sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def find_min_distance_for_4_neighbors(points):
    min_distance = float('inf')
    for point in points:
        distances = sorted(calculate_distance(point, other) for other in points if not np.array_equal(other, point))
        if len(distances) >= 4:
            min_distance = min(min_distance, distances[3])
    return min_distance if min_distance != float('inf') else None

def visualize_clusters(image, centroids, labels):
    result_image = image.copy()
    unique_labels = set(labels)
    colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(unique_labels))]
    
    for (x, y), label in zip(centroids, labels):
        if label == -1:  # Noise point
            color = (0, 0, 255)
        else:
            color = colors[label]
        cv2.circle(result_image, (int(x), int(y)), 5, color, -1)
    
    return result_image



def plot_cluster_distribution(labels):
    # Noise를 제외한 클러스터만 고려
    filtered_labels = labels[labels != -1]
    
    # 각 클러스터에 포함된 점들의 수 계산
    cluster_counts = Counter(filtered_labels)
    count_of_clusters = Counter(cluster_counts.values())
    x = list(count_of_clusters.keys())
    y = list(count_of_clusters.values())

    plt.figure(figsize=(10, 6))
    plt.bar(x, y, color=palette_hex(1))
    plt.xlabel('Number of Points in a Cluster')
    plt.ylabel('Number of Clusters')
    plt.title('Cluster Size Distribution')

    total_clusters = len(set(filtered_labels))
    total_points = len(labels)
    cluster_points = len(filtered_labels)
    cluster_percentage = (cluster_points / total_points) * 100

    plt.text(0.95, 0.95, f'Total Clusters: {total_clusters}', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor=PALETTE_NEUTRAL, edgecolor=palette_hex(0), alpha=0.8))
    plt.text(0.95, 0.90, f'Clusters Percentage: {cluster_percentage:.2f}%', horizontalalignment='right', verticalalignment='top', transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor=PALETTE_NEUTRAL, edgecolor=palette_hex(0), alpha=0.8))
    plt.show()

    
def compute_positive_area(distances, l_values):
    positive_indices = l_values > 0
    positive_distances = distances[positive_indices]
    positive_L_values = l_values[positive_indices]
    
    if len(positive_distances) > 1:
        area = simps(positive_L_values, positive_distances)
    else:
        area = 0
    return area

def calculate_pairwise_distances(points):
    return distance_matrix(points, points)

def count_points_within_distance(dist_matrix, distance, weights):
    count = 0
    for i in range(len(dist_matrix)):
        for j in range(len(dist_matrix)):
            if i != j and dist_matrix[i, j] <= distance:
                count += weights[i] * (dist_matrix[i, j] <= distance)
    return count

def is_point_in_polygon(point, polygon):
    from matplotlib.path import Path
    path = Path(polygon)
    return path.contains_point(point)

def calculate_border_weights(points, polygon, distances):
    poly = Polygon(polygon)
    weights = np.zeros((len(points), len(distances)))
    
    for i, point in enumerate(points):
        pt = Point(point)
        for j, d in enumerate(distances):
            buffer = pt.buffer(d)
            weights[i, j] = buffer.intersection(poly).area / buffer.area
    
    return weights

def compute_ripleys_L(filtered_centroids, concave_vertex, distances, polygon_area):
    num_points = len(filtered_centroids)
    dist_matrix = calculate_pairwise_distances(filtered_centroids)
    weights = calculate_border_weights(filtered_centroids, concave_vertex, distances)
    
    observed_counts = np.array([
        count_points_within_distance(dist_matrix, d, weights[:, i]) 
        for i, d in enumerate(distances)
    ])
    
    k_values = (polygon_area / num_points**2) * observed_counts
    l_values = np.sqrt(k_values / np.pi) - distances
    
    max_l_value = np.max(l_values)
    max_l_distance = distances[np.argmax(l_values)]
    
    return distances, l_values, max_l_distance, max_l_value

def plot_ripleys_L(distances, l_values, max_l_distance, max_l_value, real_distance):
    plt.plot(distances, l_values, label="Ripley's L-function")
    plt.axhline(y=0, color=palette_hex(0), linestyle='--', label="CSR (L=0)")
    plt.scatter([max_l_distance], [max_l_value], color=palette_hex(1), zorder=5)
    plt.text(max_l_distance, max_l_value, f'Max L(d)={max_l_value:.2f} at d={real_distance:.2f}', 
             fontsize=9, verticalalignment='bottom', horizontalalignment='right')
    plt.xlabel('Distance')
    plt.ylabel('L(d)')
    plt.title("Ripley's L-function")
    plt.legend()
    plt.show()

# 크기가 큰 마스크 제거 함수
def is_larger_mask(mask1, mask2):
    """Return True if mask1 is larger than mask2."""
    size1 = np.sum(mask1 > 0)
    size2 = np.sum(mask2 > 0)
    return size1 > size2

def plot_points_with_circles(filtered_centroids, concave_vertex, max_l_distance, particle_part):
    fig, ax = plt.subplots()
    
    # particle_part 이미지를 배경으로 설정
    ax.imshow(particle_part, cmap='gray', origin='lower')
    
    # Concave Hull 다각형 그리기
    polygon = plt.Polygon(concave_vertex, fill=None, edgecolor=palette_hex(0))
    ax.add_patch(polygon)
    
    # Filtered Centroids와 원 그리기
    ax.scatter(filtered_centroids[:, 0], filtered_centroids[:, 1], s=1, label="Filtered Centroids", color=palette_hex(2))
    
    for point in filtered_centroids:
        circle = Circle(point, max_l_distance, color=palette_hex(1), fill=False, linestyle='--')
        ax.add_patch(circle)
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title("Filtered Centroids with Circles at Max L Distance")
    plt.legend()
    plt.gca().invert_yaxis()  # y축 반전
    
    plt.show()


def calculate_min_distance(filtered_centroids):
    min_distances = []
    
    for i, point in enumerate(filtered_centroids):
        # 현재 점과 다른 모든 점들 사이의 거리를 계산
        distances = []
        for j, other_point in enumerate(filtered_centroids):
            if i != j:  # 자기 자신과의 거리는 제외
                distance = np.linalg.norm(point - other_point)
                distances.append(distance)
        
        # 가장 가까운 4개의 거리를 찾음
        distances.sort()
        min_radius = distances[1]  # 네 번째로 가까운 점과의 거리가 해당 점에서의 최소 반지름
        
        # 해당 점에서의 최소 반지름을 리스트에 저장
        min_distances.append(min_radius)
    
    # 모든 점에서의 최소 반지름 중 가장 작은 값을 찾음
    min_distance = min(min_distances)
    
    return min_distance

def configure_mask_generator_ui(model):
    # 매핑 테이블
    cost_time_settings = {
        0: {"points_per_side": 16, "crop_n_layers": 1, "crop_n_points_downscale_factor": 3},
        1: {"points_per_side": 32, "crop_n_layers": 1, "crop_n_points_downscale_factor": 2},
        2: {"points_per_side": 64, "crop_n_layers": 1, "crop_n_points_downscale_factor": 2}
    }
    
    data_quality_settings = {
        0: {"pred_iou_thresh": 0.95, "stability_score_thresh": 0.80},
        1: {"pred_iou_thresh": 0.95, "stability_score_thresh": 0.80},
        2: {"pred_iou_thresh": 0.95, "stability_score_thresh": 0.80}
    }
    
    hardware_spec_settings = {0: 256, 1: 256, 2: 256}
    
    # 무한 루프에서 슬라이더의 값이 변경될 때 실시간으로 파라미터 업데이트
    while True:
        # 각 슬라이더의 현재 값을 가져옴
        cost_time_level = cv2.getTrackbarPos('Cost/Time', 'Parameter Selection')
        data_quality_level = cv2.getTrackbarPos('Data Quality', 'Parameter Selection')
        hardware_spec_level = cv2.getTrackbarPos('Hardware Spec', 'Parameter Selection')
        
        # 각 슬라이더의 값에 따른 설정 매핑
        selected_cost_time = cost_time_settings[cost_time_level]
        selected_data_quality = data_quality_settings[data_quality_level]
        selected_hardware_spec = hardware_spec_settings[hardware_spec_level]
        
        # 마스크 생성기 파라미터 설정
        mask_generator = SAM2AutomaticMaskGenerator(
            model=model,
            points_per_side=selected_cost_time["points_per_side"],
            points_per_batch=selected_hardware_spec,
            pred_iou_thresh=selected_data_quality["pred_iou_thresh"],
            stability_score_thresh=selected_data_quality["stability_score_thresh"],
            crop_n_layers=selected_cost_time["crop_n_layers"],
            crop_n_points_downscale_factor=selected_cost_time["crop_n_points_downscale_factor"]
        )
        
        # 'q' 키를 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print(f"Cost/Time: {['Low', 'Medium', 'High'][cost_time_level]}, "
              f"Data Quality: {['Low', 'Medium', 'High'][data_quality_level]}, "
              f"Hardware Spec: {['Low', 'Medium', 'High'][hardware_spec_level]}")
            break
    
    # 창 닫기
    cv2.destroyAllWindows()
    
    return mask_generator

def perform_clustering_comparison(centroids, eps_value, min_samples_value, min_cluster_size=None):
    """
    Perform both DBSCAN and HDBSCAN clustering and return comparison results
    
    Args:
        centroids: array of centroid coordinates
        eps_value: DBSCAN epsilon parameter
        min_samples_value: minimum samples for both algorithms
        min_cluster_size: HDBSCAN minimum cluster size (defaults to min_samples_value)
    
    Returns:
        dict: containing results from both algorithms
    """
    centroids_array = np.array(centroids)
    
    if min_cluster_size is None:
        min_cluster_size = min_samples_value
    
    # DBSCAN clustering
    print("[INFO] Performing DBSCAN clustering...")
    dbscan = DBSCAN(eps=eps_value, min_samples=min_samples_value)
    dbscan_labels = dbscan.fit_predict(centroids_array)
    
    # HDBSCAN clustering
    print("[INFO] Performing HDBSCAN clustering...")
    hdbscan_clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, 
                                        min_samples=min_samples_value)
    hdbscan_labels = hdbscan_clusterer.fit_predict(centroids_array)
    
    # Calculate statistics for both methods
    dbscan_stats = calculate_clustering_stats(dbscan_labels, "DBSCAN")
    hdbscan_stats = calculate_clustering_stats(hdbscan_labels, "HDBSCAN")
    
    return {
        'dbscan_labels': dbscan_labels,
        'hdbscan_labels': hdbscan_labels,
        'dbscan_stats': dbscan_stats,
        'hdbscan_stats': hdbscan_stats,
        'clusterer': hdbscan_clusterer  # for additional HDBSCAN info
    }

def calculate_clustering_stats(labels, method_name):
    """Calculate statistics for clustering results"""
    unique_labels = np.unique(labels)
    n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
    n_noise = np.sum(labels == -1)
    
    cluster_sizes = []
    for label in unique_labels:
        if label != -1:
            cluster_sizes.append(np.sum(labels == label))
    
    stats = {
        'method': method_name,
        'n_clusters': n_clusters,
        'n_noise': n_noise,
        'cluster_sizes': cluster_sizes,
        'avg_cluster_size': np.mean(cluster_sizes) if cluster_sizes else 0,
        'std_cluster_size': np.std(cluster_sizes) if cluster_sizes else 0,
        'total_points': len(labels),
        'clustered_points': len(labels) - n_noise
    }
    
    return stats

def visualize_clustering_comparison(image, centroids, dbscan_labels, hdbscan_labels):
    """
    Visualize both DBSCAN and HDBSCAN clustering results side by side
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # DBSCAN visualization
    dbscan_image = visualize_clusters(image, centroids, dbscan_labels)
    axes[0].imshow(dbscan_image)
    axes[0].set_title('DBSCAN Clustering Results', fontsize=16)
    axes[0].axis('off')
    
    # HDBSCAN visualization  
    hdbscan_image = visualize_clusters(image, centroids, hdbscan_labels)
    axes[1].imshow(hdbscan_image)
    axes[1].set_title('HDBSCAN Clustering Results', fontsize=16)
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return dbscan_image, hdbscan_image

def plot_clustering_comparison_stats(dbscan_stats, hdbscan_stats):
    """
    Plot comparison statistics between DBSCAN and HDBSCAN
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Cluster count comparison
    methods = ['DBSCAN', 'HDBSCAN']
    cluster_counts = [dbscan_stats['n_clusters'], hdbscan_stats['n_clusters']]
    noise_counts = [dbscan_stats['n_noise'], hdbscan_stats['n_noise']]
    
    axes[0, 0].bar(methods, cluster_counts, color=[palette_hex(1), palette_hex(2)], alpha=0.8)
    axes[0, 0].set_title('Number of Clusters Found')
    axes[0, 0].set_ylabel('Number of Clusters')
    for i, v in enumerate(cluster_counts):
        axes[0, 0].text(i, v + 0.1, str(v), ha='center', va='bottom')
    
    axes[0, 1].bar(methods, noise_counts, color=[palette_hex(3), palette_hex(0)], alpha=0.8)
    axes[0, 1].set_title('Number of Noise Points')
    axes[0, 1].set_ylabel('Number of Noise Points')
    for i, v in enumerate(noise_counts):
        axes[0, 1].text(i, v + 0.1, str(v), ha='center', va='bottom')
    
    # Cluster size distributions
    if dbscan_stats['cluster_sizes']:
        axes[1, 0].hist(dbscan_stats['cluster_sizes'], bins=10, alpha=0.7,
                       color=palette_hex(1), label='DBSCAN')
        axes[1, 0].set_title('DBSCAN Cluster Size Distribution')
        axes[1, 0].set_xlabel('Cluster Size')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
    
    if hdbscan_stats['cluster_sizes']:
        axes[1, 1].hist(hdbscan_stats['cluster_sizes'], bins=10, alpha=0.7,
                       color=palette_hex(2), label='HDBSCAN')
        axes[1, 1].set_title('HDBSCAN Cluster Size Distribution')
        axes[1, 1].set_xlabel('Cluster Size')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def print_clustering_comparison_summary(dbscan_stats, hdbscan_stats):
    """
    Print detailed comparison summary between DBSCAN and HDBSCAN results
    """
    print("\n" + "="*80)
    print("[INFO] CLUSTERING ALGORITHM COMPARISON SUMMARY")
    print("="*80)
    
    print(f"\n{'Metric':<25} {'DBSCAN':<15} {'HDBSCAN':<15} {'Difference':<15}")
    print("-" * 70)
    print(f"{'Total Points':<25} {dbscan_stats['total_points']:<15} {hdbscan_stats['total_points']:<15} {0:<15}")
    print(f"{'Clusters Found':<25} {dbscan_stats['n_clusters']:<15} {hdbscan_stats['n_clusters']:<15} {hdbscan_stats['n_clusters'] - dbscan_stats['n_clusters']:<15}")
    print(f"{'Noise Points':<25} {dbscan_stats['n_noise']:<15} {hdbscan_stats['n_noise']:<15} {hdbscan_stats['n_noise'] - dbscan_stats['n_noise']:<15}")
    print(f"{'Clustered Points':<25} {dbscan_stats['clustered_points']:<15} {hdbscan_stats['clustered_points']:<15} {hdbscan_stats['clustered_points'] - dbscan_stats['clustered_points']:<15}")
    print(f"{'Avg Cluster Size':<25} {dbscan_stats['avg_cluster_size']:<15.2f} {hdbscan_stats['avg_cluster_size']:<15.2f} {hdbscan_stats['avg_cluster_size'] - dbscan_stats['avg_cluster_size']:<15.2f}")
    
    print(f"\nCLUSTERING EFFICIENCY:")
    dbscan_efficiency = (dbscan_stats['clustered_points'] / dbscan_stats['total_points']) * 100
    hdbscan_efficiency = (hdbscan_stats['clustered_points'] / hdbscan_stats['total_points']) * 100
    print(f"{'DBSCAN Efficiency':<25} {dbscan_efficiency:.1f}% points clustered")
    print(f"{'HDBSCAN Efficiency':<25} {hdbscan_efficiency:.1f}% points clustered")
    
    print(f"\nRECOMMENDATION:")
    if hdbscan_stats['n_clusters'] > dbscan_stats['n_clusters']:
        print("- HDBSCAN found more clusters - better for detecting fine-grained structures")
    elif hdbscan_stats['n_clusters'] < dbscan_stats['n_clusters']:
        print("- DBSCAN found more clusters - might be detecting noise as clusters")
    else:
        print("- Both methods found the same number of clusters")
        
    if hdbscan_stats['n_noise'] < dbscan_stats['n_noise']:
        print("- HDBSCAN classified fewer points as noise - more inclusive clustering")
    elif hdbscan_stats['n_noise'] > dbscan_stats['n_noise']:
        print("- HDBSCAN classified more points as noise - more conservative clustering")
    
    print("="*80)

def visualize_voronoi_and_concave_hull(vor, particle_part, concave_vertex, inside_centroids, polygons):
    """Visualize unified Voronoi diagram and concave hull."""
    fig, axes = plt.subplots(1, 3, figsize=(30, 10))
    
    # 1. Voronoi + Concave Hull 통합
    axes[0].imshow(particle_part, cmap='gray', origin='upper')
    voronoi_plot_2d(vor, ax=axes[0], show_vertices=False, line_colors=palette_hex(3), line_width=2, line_alpha=0.6, point_size=2)
    for polygon in polygons:
        x, y = polygon.exterior.xy
        axes[0].plot(x, y, color=palette_hex(0))
    for centroid in inside_centroids:
        axes[0].plot(centroid[0], centroid[1], 'o', markersize=2, color=palette_hex(1))
    for vertex in concave_vertex:
        axes[0].plot(vertex[0], vertex[1], 'o', markersize=2, color=palette_hex(2))
    axes[0].set_ylim(axes[0].get_ylim()[::-1])
    axes[0].set_title('Voronoi + Concave Hull')
    axes[0].axis('off')
    
    # 2. Voronoi만
    axes[1].imshow(particle_part, cmap='gray', origin='upper')
    voronoi_plot_2d(vor, ax=axes[1], show_vertices=False, line_colors=palette_hex(1), line_width=2, line_alpha=0.6, point_size=2)
    for centroid in inside_centroids:
        axes[1].plot(centroid[0], centroid[1], 'o', markersize=2, color=palette_hex(1))
    axes[1].set_ylim(axes[1].get_ylim()[::-1])
    axes[1].set_title('Voronoi Diagram Only')
    axes[1].axis('off')
    
    # 3. Concave Hull만
    axes[2].imshow(particle_part, cmap='gray', origin='upper')
    for polygon in polygons:
        x, y = polygon.exterior.xy
        axes[2].plot(x, y, color=palette_hex(1))
    for vertex in concave_vertex:
        axes[2].plot(vertex[0], vertex[1], 'o', markersize=2, color=palette_hex(2))
    axes[2].set_title('Concave Hull Only')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

def visualize_area_analysis(vor, boundary_pixels, concave_vertex, inside_centroids, 
                          particle_part, pixel_to_real_length=None, scale_unit="pixels"):
    """Visualize integrated area analysis (pixel and real units)."""
    
    # 면적 계산
    if pixel_to_real_length is not None:
        total_area_per_key, key_polygons_map, excluded_keys = calculate_total_area_per_key_real(
            vor, boundary_pixels, concave_vertex, inside_centroids, pixel_to_real_length)
        mask_area_map = calculate_areas_per_mask_real(boundary_pixels, pixel_to_real_length)
        areas, polygons_filtered, _, _ = calculate_filtered_areas_within_concave_hull(
            vor, boundary_pixels, concave_vertex, inside_centroids)
        # 실제 단위로 변환
        areas = [area * (pixel_to_real_length ** 2) for area in areas]
        scale_label = f"scale: {scale_unit}^2"
    else:
        total_area_per_key, key_polygons_map, excluded_keys = calculate_total_area_per_key_pixels(
            vor, boundary_pixels, concave_vertex, inside_centroids)
        mask_area_map = calculate_areas_per_mask_pixels(boundary_pixels)
        areas, polygons_filtered, _, _ = calculate_filtered_areas_within_concave_hull(
            vor, boundary_pixels, concave_vertex, inside_centroids)
        scale_label = "scale: pixel^2"
    
    # 시각화 설정
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # 1. Voronoi 영역별 면적 (Reactive Zone Areas)
    areas_array = np.array(list(total_area_per_key.values()))
    norm = Normalize(vmin=areas_array.min(), vmax=areas_array.max())
    cmap = make_value_colormap()
    
    axes[0].imshow(particle_part, cmap='gray', origin='upper')
    x, y = zip(*concave_vertex)
    axes[0].plot(x, y, color=palette_hex(1))
    
    for key, unified_polygon in key_polygons_map.items():
        area_value = total_area_per_key[key]
        color = cmap(norm(area_value))
        
        if isinstance(unified_polygon, MultiPolygon):
            for poly in unified_polygon.geoms:
                x, y = poly.exterior.xy
                axes[0].fill(x, y, color=color, alpha=0.5)
        else:
            x, y = unified_polygon.exterior.xy
            axes[0].fill(x, y, color=color, alpha=0.5)
    
    # 경계선 그리기
    for key, boundary in boundary_pixels.items():
        boundary = np.array(boundary)
        axes[0].plot(boundary[:, 0], boundary[:, 1], color=palette_hex(0), linewidth=1)
    
    axes[0].set_ylim(axes[0].get_ylim()[::-1])
    axes[0].set_title('Reactive Zone Areas')
    axes[0].axis('off')
    
    sm1 = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm1.set_array([])
    cbar1 = fig.colorbar(sm1, ax=axes[0], orientation='vertical', fraction=0.04, pad=0.04)
    cbar1.ax.tick_params(labelsize=0, length=0)
    
    # 2. 개별 입자 면적 (Particle Areas)
    mask_areas_array = np.array(list(mask_area_map.values()))
    norm2 = Normalize(vmin=mask_areas_array.min(), vmax=mask_areas_array.max())
    
    axes[1].imshow(particle_part, cmap='gray', origin='upper')
    
    for key, points in boundary_pixels.items():
        polygon = Polygon(points)
        if polygon.is_valid:
            x, y = polygon.exterior.xy
            area = mask_area_map[key]
            color = cmap(norm2(area))
            axes[1].fill(x, y, color=color, alpha=0.5)
            axes[1].plot(x, y, color=palette_hex(0), linewidth=2)
    
    axes[1].set_title('Particle Areas')
    axes[1].axis('off')
    
    sm2 = plt.cm.ScalarMappable(cmap=cmap, norm=norm2)
    sm2.set_array([])
    cbar2 = fig.colorbar(sm2, ax=axes[1], orientation='vertical', fraction=0.04, pad=0.04)
    cbar2.ax.tick_params(labelsize=0, length=0)
    
    plt.tight_layout()
    plt.show()
    
    return total_area_per_key, mask_area_map, excluded_keys

def create_area_histogram_and_violin(areas, scale_unit="pixels", title_prefix="Areas"):
    """Create the shared size and half-violin distribution views."""
    area_values = np.asarray(areas, dtype=float)
    mean_areas = float(np.mean(area_values))
    std_areas = float(np.std(area_values))
    DT = std_areas / mean_areas if mean_areas > 0 else 0.0
    AD = 1 - (1 + DT) ** -1 if mean_areas > 0 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    plot_size_distribution(
        axes[0],
        area_values,
        scale=scale_unit,
        title=f'{title_prefix} distribution',
    )
    plot_spatial_distribution(
        axes[1],
        area_values,
        scale=scale_unit,
        title=f'{title_prefix} distribution',
        value_label=title_prefix,
    )

    plt.tight_layout()
    plt.show()

    print(f"{title_prefix}:", areas)
    return mean_areas, std_areas, DT, AD

def create_integrated_dashboard(
    image: np.ndarray,
    size_data: Optional[Dict[str, Any]] = None,
    distribution_data: Optional[Dict[str, Any]] = None,
    shape_data: Optional[Dict[str, Any]] = None,
    scale: str = "nm",
    figsize: Tuple[int, int] = (20, 24)
) -> plt.Figure:
    """
    Create integrated 2x3 dashboard combining all analysis results.

    Args:
        image: Original input image (BGR)
        size_data: Dictionary containing:
            - 'areas': List or dict of particle areas
            - 'metrics': Size statistics dictionary
            - 'boundary_pixels': Boundary pixels dict for overlay
        distribution_data: Dictionary containing:
            - 'voronoi_areas': List of Voronoi area sizes (for violin plot)
            - 'voronoi_areas_dict': Dict mapping centroid to Voronoi area (for overlay)
            - 'boundary_pixels': Boundary pixels dict
        shape_data: Dictionary containing:
            - 'shapes': List of shape classifications
            - 'masks': List of SAM mask dictionaries
            - 'shape_counts': Dictionary of shape counts
            - 'color_map': Dictionary mapping shapes to RGB colors
        scale: Unit scale ("nm" or "um")
        figsize: Figure size (width, height)

    Returns:
        Matplotlib figure with 2x3 subplot grid
    """
    # Create figure with 2x3 grid
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.25)

    # Row 1: Projected particle area
    if size_data is not None:
        size_values = _first_available(size_data, 'areas_real', 'areas', default=[])
        size_metrics = _first_available(size_data, 'metrics', default={})
        size_boundary_pixels = _first_available(size_data, 'boundary_pixels', default={})
        size_area_map = _first_available(size_data, 'area_map_real', 'areas', default={})

        # Left: Size Histogram
        ax1 = fig.add_subplot(gs[0, 0])
        _plot_size_histogram(ax1, size_values, size_metrics, scale)

        # Right: Size Overlay
        ax2 = fig.add_subplot(gs[0, 1])
        if size_boundary_pixels and size_area_map:
            _plot_size_overlay(ax2, image, size_boundary_pixels, size_area_map, scale)
        else:
            _plot_placeholder(ax2, 'Scale-calibrated projected-area overlay unavailable')
    else:
        # Placeholder if projected particle area was not analyzed.
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.text(0.5, 0.5, 'Projected particle area not analyzed',
                ha='center', va='center', fontsize=14)
        ax1.axis('off')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.text(0.5, 0.5, 'Projected particle area not analyzed',
                ha='center', va='center', fontsize=14)
        ax2.axis('off')

    # Row 2: PF-SUI
    if distribution_data is not None and distribution_data.get('status') in ('unavailable', 'failed'):
        for position in (0, 1):
            ax = fig.add_subplot(gs[1, position])
            _plot_placeholder(ax, 'PF-SUI unavailable\n' + distribution_data.get('reason', 'Insufficient interior regions'))
    elif distribution_data is not None:
        spatial_values = _first_available(
            distribution_data,
            'spatial_values',
            'voronoi_areas_real',
            'voronoi_areas_list',
            'voronoi_areas',
            default=[],
        )
        voronoi_area_map = _first_available(
            distribution_data,
            'voronoi_areas_real_dict',
            'voronoi_areas_dict',
            'voronoi_areas',
            default={},
        )
        spatial_boundary_pixels = _first_available(distribution_data, 'boundary_pixels', default={})

        # Left: Violin Plot
        ax3 = fig.add_subplot(gs[1, 0])
        spatial_measure_label = _first_available(
            distribution_data,
            'spatial_measure_label',
            default='Particle-boundary-based Voronoi cell area',
        )
        is_nearest_neighbor = 'Nearest-neighbor' in spatial_measure_label
        _plot_distribution_violin(
            ax3,
            spatial_values,
            scale,
            value_label='Nearest-neighbor distance' if is_nearest_neighbor else 'Particle-boundary-based Voronoi cell area',
            unit_suffix='' if is_nearest_neighbor else '^2',
        )

        # Right: Voronoi Overlay
        ax4 = fig.add_subplot(gs[1, 1])
        if spatial_boundary_pixels and isinstance(voronoi_area_map, dict) and voronoi_area_map:
            _plot_voronoi_overlay(
                ax4,
                image,
                spatial_boundary_pixels,
                voronoi_area_map,
                scale,
                concave_vertex=_first_available(distribution_data, 'concave_vertex'),
                inside_centroids=_first_available(distribution_data, 'inside_centroids'),
            )
        else:
            _plot_placeholder(ax4, 'Particle-boundary-based Voronoi overlay unavailable')
    else:
        # Placeholder if PF-SUI was not analyzed.
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.text(0.5, 0.5, 'PF-SUI not analyzed',
                ha='center', va='center', fontsize=14)
        ax3.axis('off')

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.text(0.5, 0.5, 'PF-SUI not analyzed',
                ha='center', va='center', fontsize=14)
        ax4.axis('off')

    # Row 3: Projected morphology
    if shape_data is not None:
        shape_counts = _first_available(shape_data, 'shape_counts', default={})
        shape_color_map = _first_available(shape_data, 'color_map', default={})
        shape_masks = _first_available(shape_data, 'masks', default=[])
        shapes = _first_available(shape_data, 'shapes', default=[])

        # Left: Shape Pie Chart
        ax5 = fig.add_subplot(gs[2, 0])
        _plot_shape_pie(ax5, shape_counts, shape_color_map)

        # Right: Shape Overlay
        ax6 = fig.add_subplot(gs[2, 1])
        if shape_masks and shapes:
            _plot_shape_overlay(ax6, image, shape_masks, shapes, shape_color_map)
        else:
            _plot_placeholder(ax6, 'Projected-morphology overlay unavailable')
    else:
        # Placeholder if projected morphology was not analyzed.
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.text(0.5, 0.5, 'Projected morphology not analyzed',
                ha='center', va='center', fontsize=14)
        ax5.axis('off')

        ax6 = fig.add_subplot(gs[2, 1])
        ax6.text(0.5, 0.5, 'Projected morphology not analyzed',
                ha='center', va='center', fontsize=14)
        ax6.axis('off')

    # Add main title
    fig.suptitle('Integrated Particle Analysis Dashboard',
                 fontsize=20, fontweight='bold', y=0.995)

    return fig


def _first_available(data, *keys, default=None):
    """Return the first present non-None analysis value from a result mapping."""
    for key in keys:
        if key in data and data[key] is not None:
            return data[key]
    return default


def _plot_placeholder(ax, message):
    """Draw a consistent placeholder for an analysis component that was skipped."""
    ax.text(0.5, 0.5, message, ha='center', va='center', fontsize=14)
    ax.axis('off')


def _plot_size_histogram(ax, areas, metrics, scale):
    """Plot the standard histogram, KDE trend, and median reference."""
    del metrics  # Retained for the established dashboard call signature.
    plot_size_distribution(ax, areas, scale=scale, title='Projected-area distribution')


def _plot_size_overlay(ax, image, boundary_pixels, areas, scale):
    """Plot size heatmap overlay in subplot."""
    from matplotlib import cm

    # Create overlay
    overlay = image.copy()

    # Handle both dict and list input for areas
    if isinstance(areas, dict):
        # areas is a dict like mask_area_map: {(x,y): area_value}
        area_values = list(areas.values())
        area_dict = areas
    else:
        # areas is a list - create dict from boundary_pixels keys
        # Retain the input order while avoiding an out-of-range mismatch.
        area_dict = {key: area for key, area in zip(boundary_pixels.keys(), areas)}
        area_values = list(areas)

    # Normalize areas for colormap
    min_area = min(area_values)
    max_area = max(area_values)
    cmap = make_value_colormap()

    # Color each particle by size
    for key, pixels in boundary_pixels.items():
        if key not in area_dict:
            continue

        area = area_dict[key]
        normalized = (area - min_area) / (max_area - min_area) if max_area > min_area else 0.5
        color = cmap(normalized)[:3]
        color_bgr = tuple(int(c * 255) for c in reversed(color))

        # Draw filled mask
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        pixels_array = np.array(pixels, dtype=np.int32)
        import cv2
        cv2.fillPoly(mask, [pixels_array], 255)
        overlay[mask > 0] = cv2.addWeighted(
            overlay[mask > 0], 0.6,
            np.full_like(overlay[mask > 0], color_bgr), 0.4, 0
        )

    if len(overlay.shape) == 3 and overlay.shape[2] == 3:
        ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(overlay, cmap='gray')

    # Add area labels (positioned at top of each particle)
    from shapely.geometry import Polygon
    for key, pixels in boundary_pixels.items():
        if key not in area_dict:
            continue

        area = area_dict[key]
        polygon = Polygon(pixels)

        if polygon.is_valid:
            # Position label at top of particle
            min_y = min([point[1] for point in pixels])
            ax.text(polygon.centroid.x, min_y - 5, f'{area:.1f}',
                   color=palette_hex(0), fontsize=10, ha='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=PALETTE_NEUTRAL, alpha=0.9))

    ax.set_title('Scale-calibrated projected-area overlay', fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')

    # Add colorbar legend
    sm = cm.ScalarMappable(cmap=cmap,
                          norm=plt.Normalize(vmin=min_area, vmax=max_area))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'Area ({scale}^2)', fontsize=10, fontweight='bold')


def _plot_distribution_violin(
    ax,
    voronoi_areas,
    scale,
    value_label='Voronoi cell area',
    unit_suffix='^2',
):
    """Plot the standard half-violin raincloud with observed values and CI."""
    plot_spatial_distribution(
        ax,
        voronoi_areas,
        scale=scale,
        title='PF-SUI',
        value_label=value_label,
        unit_suffix=unit_suffix,
    )


def _plot_voronoi_overlay(
    ax,
    image,
    boundary_pixels,
    voronoi_areas,
    scale,
    concave_vertex=None,
    inside_centroids=None,
):
    """Plot Voronoi diagram overlay in subplot with area labels and colorbar."""
    from modules.distribution_analysis import compute_unified_voronoi_areas
    from shapely.geometry import MultiPolygon
    from matplotlib import cm

    # Recompute Voronoi for visualization
    _, vor, unified_regions = compute_unified_voronoi_areas(
        boundary_pixels,
        concave_vertex=concave_vertex,
        inside_centroids=inside_centroids,
    )

    # Create overlay image
    overlay = image.copy()

    # Get area range for colormap (red=large, blue=small)
    areas = list(voronoi_areas.values())
    min_area = min(areas)
    max_area = max(areas)
    cmap = make_value_colormap()

    # Color each particle's Voronoi region by area
    for key, polygon in unified_regions.items():
        if key in voronoi_areas:
            area = voronoi_areas[key]
            normalized = (area - min_area) / (max_area - min_area) if max_area > min_area else 0.5
            color = cmap(normalized)[:3]
            color_bgr = tuple(int(c * 255) for c in reversed(color))

            # Draw filled polygon
            if isinstance(polygon, MultiPolygon):
                for poly in polygon.geoms:
                    coords = np.array(poly.exterior.coords, dtype=np.int32)
                    cv2.fillPoly(overlay, [coords], color_bgr)
            else:
                coords = np.array(polygon.exterior.coords, dtype=np.int32)
                cv2.fillPoly(overlay, [coords], color_bgr)

    # Blend with original image
    overlay = cv2.addWeighted(overlay, 0.5, image, 0.5, 0)

    # Display overlay
    ax.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    # Add text labels showing Voronoi area values
    for key in boundary_pixels.keys():
        if key in voronoi_areas:
            # Calculate centroid from boundary pixels
            points = boundary_pixels[key]
            centroid_x = np.mean([p[0] for p in points])
            centroid_y = np.mean([p[1] for p in points])
            area = voronoi_areas[key]

            # Add text with white background for visibility
            ax.text(centroid_x, centroid_y, f'{area:.1f}',
                   color=palette_hex(0), fontsize=10, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=PALETTE_NEUTRAL, alpha=0.9))

    ax.set_title('Particle-boundary-based Voronoi', fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')

    # Add colorbar
    sm = cm.ScalarMappable(cmap=cmap,
                          norm=plt.Normalize(vmin=min_area, vmax=max_area))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f'Voronoi Cell Area ({scale}^2)', fontsize=10, fontweight='bold')


def _plot_shape_pie(ax, shape_counts, color_map):
    """Plot the standard shape-composition donut using the overlay palette."""
    plot_shape_composition(ax, shape_counts, color_map=color_map, title='Morphology composition')


def _plot_shape_overlay(ax, image, masks, shapes, color_map):
    """Plot shape-colored overlay in subplot with actual colored masks."""
    # Display base image
    if len(image.shape) == 3 and image.shape[2] == 3:
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(image, cmap='gray')

    # Overlay shapes with colors
    for mask, shape in zip(masks, shapes):
        segmentation = mask['segmentation']
        color = color_map.get(shape, palette_rgb(0))  # Gray for unknown

        # Create colored mask overlay
        overlay = np.zeros((*segmentation.shape, 4))
        overlay[segmentation] = [*color, 0.5]  # RGBA with alpha=0.5

        ax.imshow(overlay)

    ax.set_title('Projected-morphology overlay', fontsize=14, fontweight='bold', pad=15)
    ax.axis('off')

    # Add legend
    if color_map:
        legend_elements = [
            mpatches.Patch(facecolor=color, edgecolor=palette_hex(0), label=shape)
            for shape, color in color_map.items()
        ]
        ax.legend(handles=legend_elements, loc='upper right',
                 fontsize=10, framealpha=0.9)


def save_dashboard(fig: plt.Figure, output_path: str, dpi: int = 300):
    """
    Save dashboard figure to file.

    Args:
        fig: Matplotlib figure to save
        output_path: Path to save the figure (e.g., 'results/dashboard.png')
        dpi: Resolution in dots per inch
    """
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor=PALETTE_NEUTRAL)
    plt.close(fig)


def save_all_figures(
    output_dir: str,
    dashboard_fig: Optional[plt.Figure] = None,
    size_figs: Optional[Dict[str, plt.Figure]] = None,
    distribution_figs: Optional[Dict[str, plt.Figure]] = None,
    shape_figs: Optional[Dict[str, plt.Figure]] = None,
    dpi: int = 300
):
    """
    Save all analysis figures to output directory.

    Args:
        output_dir: Directory to save figures
        dashboard_fig: Integrated dashboard figure
        size_figs: Dictionary of size analysis figures
        distribution_figs: Dictionary of spatial uniformity analysis figures
        shape_figs: Dictionary of shape analysis figures
        dpi: Resolution in dots per inch
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save dashboard
    if dashboard_fig is not None:
        save_dashboard(dashboard_fig,
                      os.path.join(output_dir, 'integrated_dashboard.png'), dpi)

    # Save size analysis figures
    if size_figs is not None:
        for name, fig in size_figs.items():
            fig.savefig(os.path.join(output_dir, f'size_{name}.png'),
                       dpi=dpi, bbox_inches='tight', facecolor=PALETTE_NEUTRAL)
            plt.close(fig)

    # Save spatial uniformity analysis figures
    if distribution_figs is not None:
        for name, fig in distribution_figs.items():
            fig.savefig(os.path.join(output_dir, f'spatial_uniformity_{name}.png'),
                       dpi=dpi, bbox_inches='tight', facecolor=PALETTE_NEUTRAL)
            plt.close(fig)

    # Save shape analysis figures
    if shape_figs is not None:
        for name, fig in shape_figs.items():
            fig.savefig(os.path.join(output_dir, f'shape_{name}.png'),
                       dpi=dpi, bbox_inches='tight', facecolor=PALETTE_NEUTRAL)
            plt.close(fig)
