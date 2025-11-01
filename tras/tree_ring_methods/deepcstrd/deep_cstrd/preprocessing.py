from cross_section_tree_ring_detection.preprocessing import NONE, convert_center_coordinate_to_output_coordinate

def resize(im_in, height_output, width_output, cy, cx):
    from urudendro.image import resize_image_using_pil_lib

    img = resize_image_using_pil_lib(im_in, height_output, width_output)
    h_i, w_i = im_in.shape[:2]
    h_o, w_o = img.shape[:2]
    cy_output, cx_output = convert_center_coordinate_to_output_coordinate(cy, cx, h_i, w_i, h_o, w_o)
    return img, cy_output, cx_output

def preprocessing(im_in, height_output=None, width_output=None, cy=None, cx=None):
    """
    Image preprocessing steps. Following actions are made
    - image resize
    - image is converted to gray scale
    - gray scale image is equalized
    Implements Algorithm 7 in the paper.
    @param im_in: segmented image
    @param height_output: new image img_height
    @param width_output: new image img_width
    @param cy: pith y's coordinate
    @param cx: pith x's coordinate
    @return:
    - im_pre: equalized image
    - cy: pith y's coordinate after resize
    - cx: pith x's coordinate after resize
    """
    # Line 1 to 6
    if NONE in [height_output, width_output]:
        im_r, cy_output, cx_output = ( im_in, cy, cx)
    else:
        im_r, cy_output, cx_output = resize( im_in, height_output, width_output, cy, cx)

    return im_r, int(cy_output), int(cx_output)