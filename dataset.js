/**
 * Class that handles the dataset used to train an image classification model.
 */
class Dataset {
    constructor() {
      this.labels = []
    }
    
    /**
     * Adds and example image to the dataset and its respective label.
     * @param {*} example 
     * @param {*} label 
     */
    addExample(example, label) {
      if (this.xs == null) {
        this.xs = tf.keep(example);
        this.labels.push(label);
      } else {
        const oldX = this.xs;
        this.xs = tf.keep(oldX.concat(example, 0));
        this.labels.push(label);
        oldX.dispose();
      }
    }
    
    /**
     * Encodes the labels of the images to one hot encoding.
     * @param {int} numClasses 
     */
    encodeLabels(numClasses) {
      for (var i = 0; i < this.labels.length; i++) {
        if (this.ys == null) {
          this.ys = tf.keep(tf.tidy(
              () => {return tf.oneHot(
                  tf.tensor1d([this.labels[i]]).toInt(), numClasses)}));
        } else {
          const y = tf.tidy(
              () => {return tf.oneHot(
                  tf.tensor1d([this.labels[i]]).toInt(), numClasses)});
          const oldY = this.ys;
          this.ys = tf.keep(oldY.concat(y, 0));
          oldY.dispose();
          y.dispose();
        }
      }
    }
  }
  