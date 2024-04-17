// Functions to deal with button events
$(function () {
    // Preview switch
    $("a#cam-preview").bind("click", function () {
      $.getJSON("/request_preview_switch", function (data) {
        // do nothing
      });
      return false;
    });
  });
  
  $(function () {
    // Flip horizontal switch
    $("a#flip-horizontal").bind("click", function () {
      $.getJSON("/request_flipH_switch", function (data) {
        // do nothing
      });
      return false;
    });
  });
  
  $(function () {
    // Model switch
    $("a#use-model").bind("click", function () {
      $.getJSON("/request_model_switch", function (data) {
        // do nothing
      });
      return false;
    });
  });
  
  
  $(function () {
    // reset camera
    $("a#reset-cam").bind("click", function () {
      $.getJSON("/reset_camera", function (data) {
        // do nothing
      });
      return false;
    });
  });