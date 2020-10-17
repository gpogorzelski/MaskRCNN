import tensorflow as tf
import sys


def expanded_shape(orig_shape, start_dim, num_dims):
    """Inserts multiple ones into a shape vector.
    Inserts an all-1 vector of length num_dims at position start_dim into a shape.
    Can be combined with tf.reshape to generalize tf.expand_dims.
    Args:
      orig_shape: the shape into which the all-1 vector is added (int32 vector)
      start_dim: insertion position (int scalar)
      num_dims: length of the inserted all-1 vector (int scalar)
    Returns:
      An int32 vector of length tf.size(orig_shape) + num_dims.
    """
    with tf.name_scope('ExpandedShape'):
        start_dim = tf.expand_dims(start_dim, 0)  # scalar to rank-1
        before = tf.slice(orig_shape, [0], start_dim)
        add_shape = tf.ones(tf.reshape(num_dims, [1]), dtype=tf.int32)
        after = tf.slice(orig_shape, start_dim, [-1])
        new_shape = tf.concat([before, add_shape, after], 0)
        return new_shape


def meshgrid(x, y):
    """Tiles the contents of x and y into a pair of grids.
    Multidimensional analog of numpy.meshgrid, giving the same behavior if x and y
    are vectors. Generally, this will give:
    xgrid(i1, ..., i_m, j_1, ..., j_n) = x(j_1, ..., j_n)
    ygrid(i1, ..., i_m, j_1, ..., j_n) = y(i_1, ..., i_m)
    Keep in mind that the order of the arguments and outputs is reverse relative
    to the order of the indices they go into, done for compatibility with numpy.
    The output tensors have the same shapes.  Specifically:
    xgrid.get_shape() = y.get_shape().concatenate(x.get_shape())
    ygrid.get_shape() = y.get_shape().concatenate(x.get_shape())
    Args:
      x: A tensor of arbitrary shape and rank. xgrid will contain these values
         varying in its last dimensions.
      y: A tensor of arbitrary shape and rank. ygrid will contain these values
         varying in its first dimensions.
    Returns:
      A tuple of tensors (xgrid, ygrid).
    """
    with tf.name_scope('Meshgrid'):
        x = tf.convert_to_tensor(x)
        y = tf.convert_to_tensor(y)
        x_exp_shape = expanded_shape(tf.shape(x), 0, tf.rank(y))
        y_exp_shape = expanded_shape(tf.shape(y), tf.rank(y), tf.rank(x))

        xgrid = tf.tile(tf.reshape(x, x_exp_shape), y_exp_shape)
        ygrid = tf.tile(tf.reshape(y, y_exp_shape), x_exp_shape)
        new_shape = y.get_shape().concatenate(x.get_shape())
        xgrid.set_shape(new_shape)
        ygrid.set_shape(new_shape)

        return xgrid, ygrid


def _center_size_bbox_to_corners_bbox(centers, sizes):
    """Converts bbox center-size representation to corners representation.
    Args:
      centers: a tensor with shape [N, 2] representing bounding box centers
      sizes: a tensor with shape [N, 2] representing bounding boxes
    Returns:
      corners: tensor with shape [N, 4] representing bounding boxes in corners
        representation
    """
    return tf.concat([centers - .5 * sizes, centers + .5 * sizes], 1)


def _tile_anchors(grid_height,
                  grid_width,
                  scales,
                  aspect_ratios,
                  base_anchor_size,
                  anchor_stride,
                  anchor_offset):
    """Create a tiled set of anchors strided along a grid in image space.
    This op creates a set of anchor boxes by placing a "basis" collection of
    boxes with user-specified scales and aspect ratios centered at evenly
    distributed points along a grid.  The basis collection is specified via the
    scale and aspect_ratios arguments.  For example, setting scales=[.1, .2, .2]
    and aspect ratios = [2,2,1/2] means that we create three boxes: one with scale
    .1, aspect ratio 2, one with scale .2, aspect ratio 2, and one with scale .2
    and aspect ratio 1/2.  Each box is multiplied by "base_anchor_size" before
    placing it over its respective center.
    Grid points are specified via grid_height, grid_width parameters as well as
    the anchor_stride and anchor_offset parameters.
    Args:
      grid_height: size of the grid in the y direction (int or int scalar tensor)
      grid_width: size of the grid in the x direction (int or int scalar tensor)
      scales: a 1-d  (float) tensor representing the scale of each box in the
        basis set.
      aspect_ratios: a 1-d (float) tensor representing the aspect ratio of each
        box in the basis set.  The length of the scales and aspect_ratios tensors
        must be equal.
      base_anchor_size: base anchor size as [height, width]
        (float tensor of shape [2])
      anchor_stride: difference in centers between base anchors for adjacent grid
                     positions (float tensor of shape [2])
      anchor_offset: center of the anchor with scale and aspect ratio 1 for the
                     upper left element of the grid, this should be zero for
                     feature networks with only VALID padding and even receptive
                     field size, but may need some additional calculation if other
                     padding is used (float tensor of shape [2])
    Returns:
      a BoxList holding a collection of N anchor boxes
    """
    ratio_sqrts = tf.sqrt(aspect_ratios)
    heights = scales / ratio_sqrts * base_anchor_size[0]
    widths = scales * ratio_sqrts * base_anchor_size[1]

    # Get a grid of box centers
    y_centers = tf.cast(tf.range(grid_height), dtype=tf.float32)
    y_centers = y_centers * anchor_stride[0] + anchor_offset[0]
    x_centers = tf.cast(tf.range(grid_width), dtype=tf.float32)
    x_centers = x_centers * anchor_stride[1] + anchor_offset[1]
    x_centers, y_centers = meshgrid(x_centers, y_centers)

    widths_grid, x_centers_grid = meshgrid(widths, x_centers)
    heights_grid, y_centers_grid = meshgrid(heights, y_centers)
    bbox_centers = tf.stack([y_centers_grid, x_centers_grid], axis=3)
    bbox_sizes = tf.stack([heights_grid, widths_grid], axis=3)
    bbox_centers = tf.reshape(bbox_centers, [-1, 2])
    bbox_sizes = tf.reshape(bbox_sizes, [-1, 2])
    bbox_corners = _center_size_bbox_to_corners_bbox(bbox_centers, bbox_sizes)
    # return box_list.BoxList(bbox_corners)
    return bbox_corners


def generate_anchors(feature_map_shape_list,
                     scales=(0.5, 1.0, 2.0),
                     aspect_ratios=(0.5, 1.0, 2.0),
                     base_anchor_size=None,
                     anchor_stride=None,
                     anchor_offset=None):
    """ Generates anchors for specific feature map size
    Args:
      :param feature_map_shape_list: list of pairs of convnet layer resolutions in the
                    format [(height_0, width_0)].  For example, setting
                    feature_map_shape_list=[(8, 8)] asks for anchors that correspond
                    to an 8x8 layer.  For this anchor generator, only lists of length 1 are
                    allowed.
      :param scales: a list of (float) scales, default=(0.5, 1.0, 2.0)
      :param aspect_ratios: a list of (float) aspect ratios, default=(0.5, 1.0, 2.0)
      :param base_anchor_size: base anchor size as height, width (
                        (length-2 float32 list or tensor, default=[256, 256])
      :param anchor_stride: difference in centers between base anchors for adjacent
                     grid positions (length-2 float32 list or tensor,
                     default=[16, 16])
      :param anchor_offset: center of the anchor with scale and aspect ratio 1 for the
                     upper left element of the grid, this should be zero for
                     feature networks with only VALID padding and even receptive
                     field size, but may need additional calculation if other
                     padding is used (length-2 float32 list or tensor,
                     default=[0, 0])
    """
    if anchor_offset is None:
        anchor_offset = [0, 0]
    if anchor_stride is None:
        anchor_stride = [16, 16]
    if base_anchor_size is None:
        base_anchor_size = [256, 256]

    if not (isinstance(feature_map_shape_list, list)
            and len(feature_map_shape_list) == 1):
        raise ValueError('feature_map_shape_list must be a list of length 1.')
    if not all([isinstance(list_item, tuple) and len(list_item) == 2
                for list_item in feature_map_shape_list]):
        raise ValueError('feature_map_shape_list must be a list of pairs.')

    # Create constants in init_scope so they can be created in tf.functions
    # and accessed from outside of the function.
    with tf.init_scope():
        base_anchor_size = tf.cast(tf.convert_to_tensor(
            base_anchor_size), dtype=tf.float32)
        anchor_stride = tf.cast(tf.convert_to_tensor(
            anchor_stride), dtype=tf.float32)
        anchor_offset = tf.cast(tf.convert_to_tensor(
            anchor_offset), dtype=tf.float32)

    grid_height, grid_width = feature_map_shape_list[0]
    scales_grid, aspect_ratios_grid = meshgrid(scales,
                                               aspect_ratios)
    scales_grid = tf.reshape(scales_grid, [-1])
    aspect_ratios_grid = tf.reshape(aspect_ratios_grid, [-1])
    anchors = _tile_anchors(grid_height,
                            grid_width,
                            scales_grid,
                            aspect_ratios_grid,
                            base_anchor_size,
                            anchor_stride,
                            anchor_offset)

    # num_anchors = tf.shape(anchors)[0]
    # anchor_indices = tf.zeros([num_anchors])
    # tf.print(anchor_indices, output_stream=sys.stdout)
    # anchors.add_field('feature_map_index', anchor_indices)
    # anchors_dict = {
    #     'anchors': anchors,
    #     'indices': anchor_indices
    # }
    return anchors


# tf.print(generate_anchors([(50, 50)]), output_stream=sys.stdout)
