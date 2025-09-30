# (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

import numpy as np

from . import plot_color


def _sample_points_on_3d_line(start, end, num_samples=10):
    """Sample points along a 3D line segment for smooth visualization"""
    points = [start + t * (end - start) for t in np.linspace(0, 1, num_samples)]
    point_pairs = [(points[i], points[i + 1]) for i in range(len(points) - 1)]

    return point_pairs


def _box_points_to_segmented_edges(box_corners, num_segments=10):
    """
    Convert the 8 corners a 3D bounding box, to a list of sampled consecutive points on the 12 edges
    Return 12 lists of point pairs, each representing an edge
    """
    # Unpack the 8 corners of the box for clarity
    p1, p2, p3, p4, p5, p6, p7, p8 = box_corners

    # Define the 12 edges of the box as pairs of corner indices
    edge_indices = [
        (p1, p2),
        (p2, p3),
        (p3, p4),
        (p4, p1),  # Bottom face
        (p5, p6),
        (p6, p7),
        (p7, p8),
        (p8, p5),  # Top face
        (p1, p5),
        (p2, p6),
        (p3, p7),
        (p4, p8),  # Vertical edges
    ]

    # Sample points along each edge
    return [
        _sample_points_on_3d_line(start, end, num_segments)
        for start, end in edge_indices
    ]


def _check_projected_points_within_image(projected_points, image_width, image_height):
    """
    Check if the projected points are completely within the image.
    """
    for point in projected_points:
        # Found a point outside the image, return False
        if (
            point[0] < -0.5
            or point[0] > image_width - 0.5
            or point[1] < -0.5
            or point[1] > image_height
        ):
            return False
    return True


def _filter_line_segments_out_of_camera_view(
    line_segments,
    camera_calibration,
    T_World_Camera,
    image_width,
    image_height,
):
    filtered_line_segments = []
    for edge in line_segments:
        filtered_edge = []
        for seg in edge:
            start_point_in_cam = T_World_Camera.inverse() @ seg[0]
            end_point_in_cam = T_World_Camera.inverse() @ seg[1]

            # Remove if any point is behind the camera
            if start_point_in_cam[2] < 0 or end_point_in_cam[2] < 0:
                continue

            # Project the start and end points using your camera calibration
            try:
                projected_start = camera_calibration.project(start_point_in_cam)
                projected_end = camera_calibration.project(end_point_in_cam)
            except Exception:
                continue

            # Check if projection succeeded (not None)
            if projected_start is None or projected_end is None:
                continue

            # Check if either points are out of the image
            if not _check_projected_points_within_image(
                [projected_start, projected_end], image_width, image_height
            ):
                continue

            filtered_edge.append([projected_start.tolist(), projected_end.tolist()])
        filtered_line_segments.append(filtered_edge)
    return filtered_line_segments


def _compute_bbox_corners_in_world(object_dimensions, T_world_object):
    """Compute 8 corners of bounding box in world coordinates"""
    # Extract object dimensions as half extents
    half_extents = object_dimensions / 2.0
    hX, hY, hZ = half_extents[0], half_extents[1], half_extents[2]

    # Define 8 corners in object local coordinates
    corners_in_object = np.array(
        [
            [-hX, -hY, -hZ],  # Corner 0
            [hX, -hY, -hZ],  # Corner 1
            [hX, hY, -hZ],  # Corner 2
            [-hX, hY, -hZ],  # Corner 3
            [-hX, -hY, hZ],  # Corner 4
            [hX, -hY, hZ],  # Corner 5
            [hX, hY, hZ],  # Corner 6
            [-hX, hY, hZ],  # Corner 7
        ],
        dtype=np.float32,
    )  # (8, 3)

    # Transform corners to world coordinates
    corners_in_world = T_world_object @ (corners_in_object.T)
    corners_in_world_T = corners_in_world.T

    return corners_in_world_T  # (8, 3)


def project_3d_bbox_to_2d_camera(
    corners_in_world,
    T_world_camera,
    camera_calibration,
    image_width,
    image_height,
    label,
) -> tuple[list[list[float]], list, list]:
    """
    Project 3D bounding box corners to 2D camera view with visibility filtering.
    Returns:
        all_visible_line_segments (list): List of visible 2D line segments.
        all_seg_colors (list): List of colors for each segment.
        label_position (list or None): [x, y] position for label or None.
    """
    # Sample the points on the 12 edges, returns a list of 12
    line_segments_on_edges = _box_points_to_segmented_edges(corners_in_world)

    # Filter out the line segments that are not visible in the camera view
    visible_line_segments = _filter_line_segments_out_of_camera_view(
        line_segments=line_segments_on_edges,
        camera_calibration=camera_calibration,
        T_World_Camera=T_world_camera,
        image_width=image_width,
        image_height=image_height,
    )

    # Aggregate visible line segments along with colors
    all_visible_line_segments = []
    all_seg_colors = []
    plotting_colors = [
        plot_color.RED,
        plot_color.GRAY,
        plot_color.GRAY,
        plot_color.GREEN,
        plot_color.GRAY,
        plot_color.GRAY,
        plot_color.GRAY,
        plot_color.GRAY,
        plot_color.BLUE,
        plot_color.GRAY,
        plot_color.GRAY,
        plot_color.GRAY,
    ]

    for i_edge, segments in enumerate(visible_line_segments):
        for segment in segments:
            all_visible_line_segments.append(segment)
            all_seg_colors.append(plotting_colors[i_edge])

    # Calculate label position if we have visible segments and a label
    label_position = None
    if label and all_visible_line_segments:
        # Find the projected center of the bounding box for label placement
        all_x_coords = []
        all_y_coords = []

        for segment in all_visible_line_segments:
            start_point, end_point = segment
            all_x_coords.extend([start_point[0], end_point[0]])
            all_y_coords.extend([start_point[1], end_point[1]])

        if all_x_coords and all_y_coords:
            # Use the center of all projected points as label position
            center_x = sum(all_x_coords) / len(all_x_coords)
            center_y = sum(all_y_coords) / len(all_y_coords)
            label_position = [center_x, center_y]

    return all_visible_line_segments, all_seg_colors, label_position


def extract_bbox_projection_data(data_provider, evl_3d_bboxes):
    """Extract and process bounding box data for projection"""
    projection_data = []

    for instance_id, bbox3d_wrapper in evl_3d_bboxes.items():
        bbox3d_data = bbox3d_wrapper.bbox3d
        aabb = bbox3d_data.aabb

        # Calculate 3D box dimensions and get transform
        object_dimensions = np.array(
            [
                aabb[1] - aabb[0],  # width
                aabb[3] - aabb[2],  # height
                aabb[5] - aabb[4],  # depth
            ]
        )
        T_world_object = bbox3d_data.transform_scene_object

        # Get 8 corners in world coordinates - need to implement this function
        corners_world = _compute_bbox_corners_in_world(
            object_dimensions, T_world_object
        )

        # Get instance label for display
        label = f"instance_{instance_id}"
        instance_info = data_provider.get_evl_instance_info_by_id(instance_id)
        if instance_info:
            if hasattr(instance_info, "category") and instance_info.category:
                label = instance_info.category
            elif hasattr(instance_info, "name") and instance_info.name:
                label = instance_info.name

        projection_data.append(
            {"instance_id": instance_id, "corners_world": corners_world, "label": label}
        )

    return projection_data
