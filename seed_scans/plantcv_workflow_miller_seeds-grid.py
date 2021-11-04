#!/usr/bin/env python

import argparse
import os
import cv2
import numpy as np
from plantcv import plantcv as pcv
from sklearn import mixture


def options():
    """Parse command-line options."""
    parser = argparse.ArgumentParser(description='Split input images into seed images.')
    parser.add_argument("--image", help="Input directory of images.", required=True)
    parser.add_argument("--seed_count", help="Number of seeds in the image.", required=True, type=int)
    parser.add_argument("--output_dir", help="Output directory of seed images.", required=True)
    parser.add_argument("--debug", help="Save intermediate images if set to print.", default=None)
    args = parser.parse_args()
    return args


def main():
    args = options()

    # Set debug mode
    pcv.params.debug = args.debug
    pcv.params.line_thickness = 30
    pcv.params.dpi = 200

    # Open image
    img, imgpath, imgname = pcv.readimage(filename=args.image)

    sample_name = os.path.splitext(imgname)[0]

    # Create a region of interest that encloses all the seed
    roi, roi_str = pcv.roi.rectangle(img=img, x=1000, y=3500, h=15050, w=12000)

    # Sometimes the seed packet is within the region of interest
    # Use an RGB threshold to find the seed packet
    packet_mask, packet_img = pcv.threshold.custom_range(img=img, lower_thresh=[244, 178, 100], upper_thresh=[255, 190, 120], channel="rgb")

    if np.sum(packet_mask) > 0:
        # Create a whole-image ROI
        img_roi, img_str = pcv.roi.rectangle(img=img, x=0, y=0, h=img.shape[0], w=img.shape[1])

        # Dilate the edges of the seed packet to include the edges
        packet_dil = pcv.dilate(gray_img=packet_mask, ksize=9, i=13)

        # Fill in missing pixels within the seed packet binary mask
        packet_cleaned = pcv.fill_holes(bin_img=packet_dil)

        # Detect the packet contour
        packet_all_cnt, packet_all_str = pcv.find_objects(img=img, mask=packet_cleaned)

        # Find the largest (packet) ROI
        packet_cnt, packet_str, packet_bin_img, packet_area = pcv.roi_objects(img=img, roi_contour=img_roi, roi_hierarchy=img_str, object_contour=packet_all_cnt, obj_hierarchy=packet_all_str, roi_type="largest")

        # Combine the contours into one object
        packet_obj, _ = pcv.object_composition(img=img, contours=packet_cnt, hierarchy=packet_str)

        # Fit an upright bounding box
        packet_x, packet_y, packet_w, packet_h = cv2.boundingRect(packet_obj)

        # Create a an empty mask
        packet_box_mask = np.zeros(packet_cleaned.shape, dtype=np.uint8)

        # Draw the bounding box on the mask
        cv2.rectangle(packet_box_mask, (packet_x, packet_y), (packet_x + packet_w, packet_y + packet_h), (255), -1)

        # Invert the packet mask
        img_mask = pcv.invert(gray_img=packet_box_mask)

        # Apply the inverted packet mask to the original image to remove the seed packet
        img_no_packet = pcv.apply_mask(img=img, mask=img_mask, mask_color="white")
    else:
        img_no_packet = np.copy(img)

    # Create a mask from the ROI
    roi_mask = pcv.roi.roi2mask(img=img, contour=roi)

    # Mask the image so we only have the seed area left
    img_seeds_only = pcv.apply_mask(img=img_no_packet, mask=roi_mask, mask_color="white")

    # Convert RGB image to grayscale using LAB "L" channel
    gray_img = pcv.rgb2gray_lab(rgb_img=img_seeds_only, channel="l")

    # Find seed markers by applying an imprecise threshold
    # We just need the approximate location of each seed
    seeds = pcv.threshold.otsu(gray_img=gray_img, max_value=255, object_type="dark")

    # Blur horizontal background structure from top/bottom of labels/rulers
    blurred = pcv.median_blur(gray_img=seeds, ksize=(33, 1))

    # Remove any small spots in the background
    seeds_remove_small = pcv.fill(bin_img=blurred, size=1000)

    # Dilate the seeds to fill them in
    labels = pcv.dilate(gray_img=seeds_remove_small, ksize=9, i=5)

    # Fill in gaps in the seed labels
    labels_filled = pcv.fill_holes(bin_img=labels)

    # Detect markers for each seed
    markers, marker_str = pcv.find_objects(img=img, mask=labels_filled)

    # Determine the size of the "seeds"
    marker_sizes = []
    for marker in markers:
        # Calculate the pixel area of the contour
        area = cv2.contourArea(marker)
        marker_sizes.append(area)
    sorted_markers = np.argsort(-np.array(marker_sizes))
    filtered_markers = []
    for i in range(args.seed_count):
        filtered_markers.append(markers[sorted_markers[i]])

    # Loop over each seed marker
    seed_imgs = []
    seed_cols = []
    seed_rows = []
    n_markers = len(filtered_markers)
    n_cols = int(n_markers / 12)
    print(f"Image {sample_name}: expected {args.seed_count} seeds and found {n_markers} seeds in {n_cols} columns")
    for marker in filtered_markers:
        # Calculate the moments for each contour
        M = cv2.moments(marker)
        # Centroid x coordinate
        cx = int(M["m10"] / M["m00"])
        # Centroid y coordinate
        cy = int(M["m01"] / M["m00"])
        # Append the x coordinate to the column list
        seed_cols.append(cx)
        # Append the y coordinate to the row list
        seed_rows.append(cy)
        # Set padding around each seed to 100 pixels
        padding = 100
        # Find the minimum upright bounding rectangle
        x, y, width, height = cv2.boundingRect(marker)
        # Set left, right, up, down padding
        x1 = x - padding
        y1 = y - padding
        if x1 < 0:
            x1 = 0
        if y1 < 0:
            y1 = 0
        x2 = x1 + width + padding * 2
        y2 = y1 + height + padding * 2
        # Extract the seed region from the input RGB image
        seed_img = img_no_packet[y1:y2, x1:x2, :]
        # Append to a list of seed images
        seed_imgs.append(seed_img)

    # Fit a Guassian Mixture Model on the x-axis coordinates assuming that there are n columns
    # The n GMM means are the centers of each column
    # Create a GMM object
    gmm = mixture.GaussianMixture(n_components=n_cols, covariance_type="full")
    # Train the model with x-coordinates
    gmm.fit(np.array(seed_cols).reshape(-1, 1))
    # Sort the columns by x-coordinate and determine the column IDs
    col_order = np.argsort(gmm.means_.reshape(1, -1)[0])

    # Initialize the columns
    columns = {}
    for i, col in enumerate(col_order):
        # Each column refers to a group in the GMM
        columns[col] = i
    # Initialize the column groupings
    column_grps = {}
    for i in range(n_cols):
        column_grps[i] = []
    # Predict which column each seed belongs to using the x-coordinate
    pred = gmm.predict(np.array(seed_cols).reshape(-1, 1))
    # For each seed, store it's ID in the predicted column
    for i, grp in enumerate(pred):
        column_grps[columns[grp]].append(i)
    # Store row groupings
    row_grps = {}
    for i in range(n_cols):
        row_pos = []
        for seed_id in column_grps[i]:
            # Append the y-coordinate to each column group
            row_pos.append(seed_rows[seed_id])
        # Sort the seeds per column by y-coordinate (row)
        sorted_row_ids = np.argsort(row_pos)
        # For each row group, store the overall seed ID
        # Note that this is the ID from OpenCV which is labeled from
        # The bottom of the image up
        row_grps[i] = [column_grps[i][j] for j in sorted_row_ids]

    # Number the seeds from 1 to n from the top of the first column
    # on the left to the bottom of the last column on the right
    seed_ids = {}
    seed_id = 1
    # Loop over each column left to right
    for i in range(n_cols):
        # Loop over each row from top to bottom
        for j in row_grps[i]:
            # Store the seed ID
            seed_ids[j] = seed_id
            # Increment the ID
            seed_id += 1

    # Create the output directory if it does not exist
    os.makedirs(args.output_dir, exist_ok=True)
    # Iterate over each seed and store the seed image in a file
    for i, seed_img in enumerate(seed_imgs):
        pcv.print_image(img=seed_img, filename=os.path.join(args.output_dir, f"{sample_name}_{seed_ids[i]}.png"))


if __name__ == "__main__":
    main()
