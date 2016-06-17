#seed_size_plantcv_analysis.py

#Imports
import argparse
import posixpath
import numpy as np
import cv2
import plantcv as pcv
from plantcv.dev.color_palette import color_palette

def options():
    parser = argparse.ArgumentParser(description="Imaging processing with opencv")
    parser.add_argument("-i", "--image", help="Input image file.", required=True)
    parser.add_argument("-o", "--outdir", help="Output directory for image files.", required=True)
    parser.add_argument("-D", "--debug", help="Turn on debug, prints intermediate images.", default=None)
    parser.add_argument("-w", "--writeimg", help="Write out images to file.", action="store_true")
    parser.add_argument("-r", "--result", help="Output result file", required=True)
    args = parser.parse_args()
    return args

def main():

    # Sets variables from input arguments
    args = options()

    device = 0                                       # Workflow step counter
    debug = args.debug                               # Option to display debug images to the notebook
    rgb_img = args.image                             # Name of seed Image
    writeimg = args.writeimg
    outfile = args.result
    outdir = args.outdir

    # Reads in RGB image and plots using pyplot
    img, path, filename = pcv.readimage(rgb_img)
    if writeimg == True:
        pcv.print_image(img, (outfile + "_initimg"))

    # Converts RGB to HSV and extract the Saturation channel
    device, img_gray_sat = pcv.rgb2gray_hsv(img, 's', device, debug)

    # Thresholds the Saturation image and saves as binary image
    device, img_binary = pcv.binary_threshold(img_gray_sat, 25, 255, 'light', device, debug)

    # Fills in speckles smaller than 150 pixels
    mask = np.copy(img_binary)
    device, fill_image= pcv.fill(img_binary, mask, 150, device, debug)

    # Identifies objects using filled binary image as a mask
    device, id_objects, obj_hierarchy = pcv.find_objects(img, fill_image, device, debug)

    # Defines rectangular ROI
    device, roi, roi_hierarchy = pcv.define_roi(img, 'rectangle', device, None, 'default', debug, True, 300, 1000, -1250, -425)

    # Keeps only objects within or partially within ROI
    device, roi_objects, roi_obj_hierarchy, kept_mask, obj_area = pcv.roi_objects(img, 'partial', roi, roi_hierarchy,
                                                                               id_objects, obj_hierarchy, device,debug)

    # Randomly colors the individual seeds
    img_copy = np.copy(img)
    for i in range(0, len(roi_objects)):
        rand_color = color_palette(1)
        cv2.drawContours(img_copy, roi_objects, i, rand_color[0], -1, lineType=8, hierarchy=roi_obj_hierarchy)
    if writeimg == True:
        pcv.print_image(img, (outfile + "_coloredseeds"))

    # Gets the area of each seed, saved in shape_data
    shape_header = []
    table = []
    for i in range(0, len(roi_objects)):
        if roi_obj_hierarchy[0][i][3] == -1: #Checks if shape is a parent contour

            # Object combine kept objects
            device, obj, mask2 = pcv.object_composition(img, [roi_objects[i]], np.array([[roi_obj_hierarchy[0][i]]]),
                                                   device, debug)
            if obj is not None:
                device, shape_header, shape_data, shape_img = pcv.analyze_object(img, rgb_img, obj, mask2, device, debug)
                if shape_data is not None:
                    table.append(shape_data)
                    #print(shape_data[1])      #Prints area to screen


    # Finds the area of the size marker in pixels and saves to "marker data"
    device, marker_header, marker_data, analysis_images = pcv.report_size_marker_area(img, 'rectangle', device, debug,
                                                                                          "detect", 3525, 850, -200, -1700, "black", 'light', 'h',
                                                                                          120)
    shape_header.append("marker_area")

    # Saves seed and marker shape data results to file
    results = open(posixpath.join(outdir, outfile), 'w')
    results.write('\t'.join(map(str, shape_header)) + '\n')
    for row in table:
        row.append(marker_data[1])
        results.write('\t'.join(map(str, row)) + '\n')
    results.close()

if __name__ == '__main__':
    main()