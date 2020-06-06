/**
 * Gets an image as input and normalizes it between -1 and 1.
 * Returns a batched image (1-element batch) of shape [1, w, h, c].
 * @param {HTMLImageElement} img An image element.
 */
function convertToTensor4D(img) {
    return tf.tidy(() => {
    // Reads the image as a Tensor from the image element.
    const tensorImage = tf.browser.fromPixels(img);

    resized = tf.image.resizeBilinear(tensorImage, [224,224]);

    // Expand the outer most dimension so we have a batch size of 1.
    const batchedImage = resized.expandDims(0);

    // Normalize the image between -1 and 1. The image comes in between 0-255,
    // so we divide by 127 and subtract 1.
    return batchedImage.toFloat().div(tf.scalar(127)).sub(tf.scalar(1));
    });
}

/**
 * Crops an image tensor so we get a square image with no white space.
 * @param {Tensor4D} img An input image Tensor to crop.
 */
function cropImage(img) {
    const size = Math.min(img.shape[0], img.shape[1]);
    const centerHeight = img.shape[0] / 2;
    const beginHeight = centerHeight - (size / 2);
    const centerWidth = img.shape[1] / 2;
    const beginWidth = centerWidth - (size / 2);
    return img.slice([beginHeight, beginWidth, 0], [size, size, 3]);
}