
var word_current = "------------";
var category_current = "----------";
    

let chunks = [];

function onVideoFail(e) {
  console.log('webcam fail!', e);
};

function hasGetUserMedia() {
  // Note: Opera is unprefixed.
  return !!(navigator.getUserMedia || navigator.webkitGetUserMedia ||
    navigator.mozGetUserMedia || navigator.msGetUserMedia);
}

if (hasGetUserMedia()) {
  // Good to go!
} else {
  alert('getUserMedia() is not supported in your browser');
}

window.URL = window.URL || window.webkitURL;
navigator.getUserMedia = navigator.getUserMedia ||
  navigator.webkitGetUserMedia ||
  navigator.mozGetUserMedia ||
  navigator.msGetUserMedia;

var video = document.querySelector('#webCamera');
var webcamstream;
var streamRecorder;
var recording = false;

if (navigator.getUserMedia) {
  navigator.mediaDevices.getUserMedia({ audio: false, video: true }).then(function (stream) {
    video.srcObject = stream;
    webcamstream = stream;
    streamRecorder = new MediaRecorder(stream);


    function Record() {
      button = document.getElementById("recordButton")
      if (recording) {
        button.className = "btn btn-outline-success btn-lg mb-3";
        button.innerText = "Записать";
        stopRecording();
      }
      else {
        button.className = "btn btn-outline-danger btn-lg mb-3";
        button.innerText = "Остановить запись";
        document.getElementById("resultSpan").innerText = "";
        startRecording();
      }
    }

    document.getElementById("recordButton").onclick = (e) => {
      Record();
    };
    
    $(document).bind('keydown',function(e){
      key  = e.keyCode;
      if(key == 32){
          Record();
      }
  });

    function startRecording() {
      streamRecorder.start();
      recording = true;
    }
    function stopRecording() {
      streamRecorder.stop();
      recording = false
    }

    streamRecorder.onstop = (e) => {
      const blob = new Blob(chunks, { type: "video\/mp4" });
      chunks = [];
      console.log("recorder stopped");
      postVideoToServer(blob);
    };

    streamRecorder.ondataavailable = (e) => {
      chunks.push(e.data);
    };

    function postVideoToServer(videoblob) {
      var data = new FormData();
      data.append("video", videoblob, "filename" + Date.now() + ".mp4");
      data.append("category", category_current);
      $.ajax({
        url: "/submit/",
        type: 'POST',
        data: data,
        processData: false,
        contentType: false,
        success: function (data) {
          console.log(data);
          document.getElementById("resultSpan").innerText = "Распознано: " + data;
        }
      });
    }
  }).catch(onVideoFail);
} else {
  alert('failed');
}


function showGuide() {
  $('#guideVideo').toggle();
  //guideVideo = document.getElementById("guideVideo")
  //guideVideo.hidden = !guideVideo.hidden;
}

function showhideWords() {
  $('.word-list').toggle();
}

function getCategory(elem) {
  category = elem.id;
  console.log(category);
  getTask(word=null, category=category);
}

function getWord(elem) {
  word = elem.id;
  console.log(word);
  getTask(word=word, category=null);
}

function getRandom() {
  console.log("random");
  getTask();
}

function getTask(word=null, category=null) {
  $.ajax({
    url: "/getTask/",
    type: "get", //send it through get method
    data: { 
      word: word,
      category: category
    },
    success: function(response) {
      word_current = response.word;
      category_current = response.category;
      $('#taskWord').contents().first().replaceWith('Слово: ' + word_current);
      $('#guideVideo').attr('src', response.guide_path);
      $('#guideVideo').load();
      console.log(word);
    },
    error: function(xhr) {
      //Do Something to handle error
    }
  });

  // list-group-item-success
  
}