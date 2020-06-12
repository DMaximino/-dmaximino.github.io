// Progress bar initialization
let uploadProgress = []
let progressBarClassA = document.getElementById('progress-bar-class-a')
let progressBarClassB = document.getElementById('progress-bar-class-b')
let progressBarTest = document.getElementById('progress-bar-test')

/**
 * Initialize progress bar.
 * @param {Int} numFiles 
 * @param {*} progressBar 
 */
function initializeProgress(numFiles, progressBar) {
  progressBar.value = 0
  progressBar.max = numFiles
}

/**
 * Updates the progress bar on the interface.
 * @param {*} progressBar Progress bar html element.
 */
function updateProgress(progressBar) {
  progressBar.value = progressBar.value + 1
}


// Drag and drop initialization
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

/**
 * Prevents defaults.
 * @param {*} e Data of the event.
 */
function preventDefaults(e) {
  e.preventDefault()
  e.stopPropagation()
}

/**
 * Adds highlight on interface.
 * @param {*} e Data of the event.
 */
function highlight(e) {
  document.getElementById(e.target.id).classList.add('highlight')
}

/**
 * Removes highlight on interface.
 * @param {*} e Data of the event.
 */
function unhighlight(e) {
  document.getElementById(e.target.id).classList.remove('highlight')
}

/**
 * Handles the file dropping.
 * @param {*} e Data of the droppend content.
 */
function handleDrop(e) {
  var dt = e.dataTransfer
  var files = dt.files

  handleFiles(files, e.target.id)
}

//TODO: Make scalable. Avoid code repetition.
/**
 * Handles files dropped on interface.
 * @param {Array} files Files that represents images.
 * @param {String} id 
 */
async function handleFiles(files, id) {
  files = [...files]

  if (id == 'drop-area-class-a') {
    initializeProgress(files.length, progressBarClassA)
    files.forEach(previewAndUploadFileClassA)
  }
  else if (id == 'drop-area-class-b') {
    initializeProgress(files.length, progressBarClassB)
    files.forEach(previewAndUploadFileClassB)
  }
  else if (id == 'drop-area-test') {
    initializeProgress(files.length, progressBarTest)
    files.forEach(previewFileAndPredict)
  }
}

/**
 * Previews and uploads files of class A.
 * @param {*} file File that represents an image.
 * @param {Int} i 
 * @param {Array} arr 
 */
function previewAndUploadFileClassA(file, i, arr) {
  previewFile(file, i, arr, document.getElementById('gallery-class-a'))
  uploadFile(file, 0, updateProgress, progressBarClassA)
}

/**
 * Previews and uploads files of class B.
 * @param {*} file File that represents an image.
 * @param {Int} i 
 * @param {Array} arr 
 */
function previewAndUploadFileClassB(file, i, arr) {
  previewFile(file, i, arr, document.getElementById('gallery-class-b'))
  uploadFile(file, 1, updateProgress, progressBarClassB)
}

/**
 * Reads file as an image and shows its preview on interface.
 * @param {*} file File that represents an image.
 * @param {Int} i 
 * @param {Array} arr 
 * @param {String} id Id of the drop area where to show the images.
 */
function previewFile(file, i, arr, id) {
  let reader = new FileReader()
  reader.readAsDataURL(file)
  reader.onloadend = function () {
    let img = document.createElement('img')
    img.src = reader.result
    id.appendChild(img)
  }
}

/**
 * Reads a file as an image and adds to the dataset. 
 * A callback is sent to update the progress bar on the interface.
 * @param {*} file File that represents an image.
 * @param {Int} label Label of the image (class).
 * @param {function updateProgress(id) {
   
 }} callback Callback function, currently used to update the progress bar.
 */
function uploadFile(file, label, callback) {
  args = arguments;
  let reader = new FileReader()
  reader.readAsDataURL(file)
  reader.onloadend = function () {
    var img = document.createElement('img');
    img.src = reader.result
    //document.body.appendChild(img);
    img.onload = function () {
      addExampleToDataset(img, label)
      callback(args[3]);
    }
  }

}

/**
 * Reads a file as an image and its preview to the interface, then
 * performs prediction on the image based on the trained classifier model.
 * @param {*} file File that represents an image.
 */
async function previewFileAndPredict(file) {

  let reader = new FileReader()
  reader.readAsDataURL(file)
  reader.onloadend = async function () {
    let img = document.createElement('img')
    img.src = reader.result
    img.onload = async function () {
      const classId = await predict(img)
      var predictionText = "";
      switch (classId) {
        case 0:
          predictionText = document.getElementById("myInputA").value;
          break;
        case 1:
          predictionText = document.getElementById("myInputB").value;
          break;

      }
      let id = document.getElementById('gallery-test');
      var pred = document.createElement("P");
      var t = document.createTextNode(predictionText);
      pred.appendChild(t);
      var canvas = document.createElement('canvas');
      canvas.width = 150;
      canvas.height = 150;
      var ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, img.width, img.height, 0, 0, 150, 150);
      ctx.strokeStyle = 'white';
      ctx.font = "20px Arial";
      ctx.strokeText(predictionText, 5, 130)
      ctx.fillText(predictionText, 5, 130)
      id.appendChild(canvas);
      updateProgress(progressBarTest);
    }
  }
}


// Training interface

/**
 * Enables train button and changes its text.
 */
function enableTrainButton() {
  document.getElementById("train").innerText = "Train your classifier!";
  enableButton("train");
}

/**
 * Disables train button and changes its text.
 */
function disableTrainButton() {
  document.getElementById("train").innerText = "Trainning...";
  disableButton("train");
}

/**
 * Enables a button.
 * @param {String} buttonId Button element id.
 */
function enableButton(buttonId) {
  document.getElementById(buttonId).disabled = false;
}

/**
 * Disables a button.
 * @param {String} buttonId Button element id.
 */
function disableButton(buttonId) {
  document.getElementById(buttonId).disabled = false;
}

/**
 * Callback to run when train ends.
 * Changes in interface.
 */
function callbackTrainnigEnd() {
  enableTrainButton();
  enableButton("saveModel");
  $(document).ready(function(){
    $("#myToast").toast('show');
  });
 
}

/**
 * Starts the training and updates the interface accordingly.
 */
function doTraining() {
  if (areExamplesInDataset() == false)
    return

  disableTrainButton();
  train(callbackTrainnigEnd);

}

// Download model interface

function doDownloadModel() {
  saveModel();
}

