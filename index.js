const dataset = new Dataset();
let model = null
let mobilenet;

async function loadMobilenet() {
  const mobilenet = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_1.0_224/model.json');
  const layer = mobilenet.getLayer('conv_pw_13_relu');
  return tf.model({inputs: mobilenet.inputs, outputs: layer.output});
}

async function train() {
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
          document.getElementById("train").innerText = "Train your classifier!";
          document.getElementById("train").disabled = false;
        }
      }
   });
}

async function doPredict(){
  testImage = document.getElementById("testImage");
  const classId = await predict(testImage);
  var predictionText = "";
  switch(classId){
		case 0:
			predictionText = document.getElementById("myInputA").value;
			break;
		case 1:
			predictionText = document.getElementById("myInputB").value;
			break;
                
  }
  console.log("prediction")
  
	document.getElementById("pred").innerText = predictionText;
}

/**
 * 
 */
function doTraining(){
  //TODO: Check whether the dataset has elements
  document.getElementById("train").disabled = true;
  document.getElementById("train").innerText = "Trainning...";
  train();
  document.getElementById("saveModel").disabled = false;
}

/**
 * 
 * @param {} img 
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


async function saveModel()
{
  model.save('downloads://classifier');
} 

//Disable download model
document.getElementById("saveModel").disabled = true;

// ************************ Drag and drop ***************** //
let dropAreaClassA = document.getElementById("drop-area-class-a")
let dropAreaClassB = document.getElementById("drop-area-class-b")

// Prevent default drag behaviors
;['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
  dropAreaClassA.addEventListener(eventName, preventDefaults, false)
  dropAreaClassB.addEventListener(eventName, preventDefaults, false)    
  document.body.addEventListener(eventName, preventDefaults, false)
})

// Highlight drop area when item is dragged over it
;['dragenter', 'dragover'].forEach(eventName => {
  dropAreaClassA.addEventListener(eventName, highlight, false)
  dropAreaClassB.addEventListener(eventName, highlight, false)
})

;['dragleave', 'drop'].forEach(eventName => {
  dropAreaClassA.addEventListener(eventName, unhighlight, false)
  dropAreaClassB.addEventListener(eventName, unhighlight, false)
})

// Handle dropped files
dropAreaClassA.addEventListener('drop', handleDrop, false)
dropAreaClassB.addEventListener('drop', handleDrop, false)

function preventDefaults (e) {
  e.preventDefault()
  e.stopPropagation()
}

function highlight(e) {
  document.getElementById(e.target.id).classList.add('highlight')
}

function unhighlight(e) {
  document.getElementById(e.target.id).classList.remove('highlight')
}

function handleDrop(e) {
  var dt = e.dataTransfer
  var files = dt.files

  handleFiles(files, e.target.id)
}

let uploadProgress = []
let progressBarClassA = document.getElementById('progress-bar-class-a')
let progressBarClassB = document.getElementById('progress-bar-class-b')
let progressBarTest = document.getElementById('progress-bar-test')

function initializeProgress(numFiles, progressBar) {
  progressBar.value = 0
  progressBar.max = numFiles
}

function updateProgress(progressBar) {
  progressBar.value = progressBar.value + 1
}

async function handleFiles(files, id) {
  files = [...files]


  if(id == 'drop-area-class-a'){
    initializeProgress(files.length, progressBarClassA)
    files.forEach(uploadFileClassA)
    files.forEach(previewFileClassA)
  }
  else if(id == 'drop-area-class-b'){
    initializeProgress(files.length, progressBarClassB)
    files.forEach(uploadFileClassB)
    files.forEach(previewFileClassB)
  }
  else if(id =='drop-area-test'){
    initializeProgress(files.length, progressBarTest)
    files.forEach(previewFileAndPredict)
  }
}

function previewFileClassA(file, i, arr){
  previewFile(file, i, arr, document.getElementById('gallery-class-a'))
}

function previewFileClassB(file, i, arr){
  previewFile(file, i, arr, document.getElementById('gallery-class-b'))
}

function previewFile(file, i, arr, id) {
  let reader = new FileReader()
  reader.readAsDataURL(file)
  reader.onloadend = function() {
    let img = document.createElement('img')
    img.src = reader.result
    id.appendChild(img)
  }
}

async function previewFileAndPredict(file, i, arr) {

  let reader = new FileReader()
  reader.readAsDataURL(file)
  reader.onloadend = async function() {
    let img = document.createElement('img')
    img.src = reader.result
    img.onload = async function () {
      const classId = await predict(img)
      var predictionText = "";
      switch(classId){
        case 0:
          predictionText = document.getElementById("myInputA").value;
          break;
        case 1:
          predictionText = document.getElementById("myInputB").value;
          break;
                    
      }
      let id = document.getElementById('gallery-test');
      var pred = document.createElement("P");                       // Create a <p> node
      var t = document.createTextNode(predictionText);      // Create a text node
      pred.appendChild(t);                                          // Append the text to <p>
      var canvas = document.createElement('canvas');
      canvas.width = 150;
      canvas.height = 150;
      var ctx = canvas.getContext('2d');
      ctx.drawImage(img,0,0,img.width,img.height,0,0,150,150);
      ctx.strokeStyle = 'white';
      ctx.font = "20px Arial";
      ctx.strokeText(predictionText, 5,130)
      ctx.fillText(predictionText, 5,130)
      id.appendChild(canvas);
      updateProgress(progressBarTest);  
    }
  }
}

function uploadFileClassA(file, i){
  uploadFile(file, 0, updateProgress, progressBarClassA)
}

function uploadFileClassB(file, i){
  uploadFile(file, 1, updateProgress, progressBarClassB)
}

function uploadFile(file, label, callback) {
  args = arguments;
  let reader = new FileReader()
  reader.readAsDataURL(file)
  reader.onloadend = function() {
    var img = document.createElement('img');
    img.src = reader.result
    //document.body.appendChild(img);
    img.onload = function () {
      const tensor = convertToTensor4D(img)
      dataset.addExample(mobilenet.predict(tensor), label);
      callback(args[3]);
    } 
  }

}

async function init(){
  mobilenet = await loadMobilenet();		
  //alert("mobilenet loaded!")
}


init();