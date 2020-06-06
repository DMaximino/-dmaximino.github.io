// Initialize variables
const dataset = new Dataset();
let model = null
let mobilenet;

/**
 * Loads mobilenet model with pretrained weights.
 */
async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

/**
 * Performs the training using the initialized model mobilenet and the dataset.
 */
async function train(callback) {
  dataset.ys = null;
  dataset.encodeLabels(2);
    
  model = tf.sequential({
    layers: [
      tf.layers.flatten({inputShape: mobilenet.outputs[0].shape.slice(1)}),
      tf.layers.dense({ units: 100, activation: 'relu'}),
      tf.layers.dense({ units: 2, activation: 'softmax'})
    ]
  });
    
   
  // Set the optimizer to be tf.train.adam() with a learning rate of 0.0001.
  const optimizer = tf.train.adam(0.0001);
    
        
  // Compile the model using the categoricalCrossentropy loss, and
  // the optimizer you defined above.
  model.compile({optimizer: optimizer, loss: 'categoricalCrossentropy'});
 
  let loss = 0;
  model.fit(dataset.xs, dataset.ys, {
    epochs: 10,
    callbacks: {
      onBatchEnd: async (batch, logs) => {
        loss = logs.loss.toFixed(5);
        console.log('LOSS: ' + loss);
        },
        onTrainEnd: async (logs) => {
          alert("Trainning complete!");
          callback();
        }
      }
   });
}

/**
 * Predicts the class of an image using the custom trained model.
 * @param {HTMLImageElement} img Image HTML element.
 */
async function predict(img) {

    const predictedClass = tf.tidy(() => {
      const tensor = convertToTensor4D(img);
      const activation = mobilenet.predict(tensor);
      const predictions = model.predict(activation);
      return predictions.as1D().argMax();
    });
    const classId = (await predictedClass.data())[0];
    
      
    predictedClass.dispose();
  
    return classId
  }

  
/**
 * Starts the download of the trained model. 
 * There are 2 files downloaded classifier.weights.bin and classifier.json.
 */
async function saveModel()
{
  model.save('downloads://classifier');
} 

/**
 * Adds an example to the dataset.
 * @param {*} img Image html element.
 * @param {*} label Label or class of the image.
 */
function addExampleToDataset(img, label) {
    const tensor = convertToTensor4D(img);
    dataset.addExample(mobilenet.predict(tensor), label);
}

/**
 * Check wheter there are elements in the dataset.
 */
function areExamplesInDataset(){
    return dataset.labels.length > 0
}

/**
 * Initializes mobilenet model.
 */
async function init(){
    mobilenet = await loadMobilenet();		
    //alert("mobilenet loaded!")
}
  
  
init();