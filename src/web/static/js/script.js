'use strict';

let es = new EventSource('/stream');
es.onmessage = (event) => {
  //console.log("onmessage:", event);
};

es.onerror = (event) => {
  //console.log("onerror:" + event);
  es.close();
}

es.addEventListener("flask-event", (event) => {
  //console.log("event map length:", singleton.map.size);
  if ((event === null) || (event === "")) {
    return;
  }

  let taskId = event.data;
  singleton.map.delete(taskId);
  //console.log("event map length:", singleton.map.size);

  // task list 제거
  $("#title-" + taskId).remove();
  $("#detail-" + taskId).remove();

  // remove badge.
  if (singleton.map.size == 0) {
    $("#btn-show-task-list > span").remove();
  }
}, false);

$(document).ready(() => {

  // 첫 페이지 설정.
  InitializePage(PageName.TrainPredict);

  $("#nav-ml").on("click", function () {
    InitializePage(PageName.TrainPredict);
  });

  $("#nav-insert-data").on("click", function () {
    InitializePage(PageName.InsertData);
  });

  $("#nav-train-predict").on("click", function () {
    InitializePage(PageName.TrainPredict);
  });
});

window.onbeforeunload = function (event) {
  if (typeof event == "undefined") {
    event = window.event;
  }

  if (event) {
    es.close();
  }
};

function BtnAddClick(clickedPageName) {

  if (!singleton.id) {
    return false;
  }

  // task 개수 제한.
  let $taskList = $("footer a");
  if ($taskList.length >= 10) {
    //alert("10개 이상 추가 불가능");
    return false;
  }

  // 유효성 체크 실패한 항목이 있으면 해당 항목의 첫번째 항목에 포커싱.
  let jqueryPageId = StringFormat("[id='%%']", clickedPageName);
  let invalidPageId = StringFormat("%% .is-invalid", jqueryPageId);
  if ($(invalidPageId).length > 0) {
    $(invalidPageId).eq(0).focus();
    return false;
  }

  let btnTitle = clickedPageName;

  switch (clickedPageName) {
    case PageName.InsertData:
    case PageName.InsertTable:
      // 2개 페이지는 모든 값이 존재하는지 체크.
      let findElements = StringFormat("%% select, %% input",
        jqueryPageId, jqueryPageId);
      if (!CheckElementEmptyValue($(findElements))) {
        return false;
      }
      break;

    case PageName.TrainPredict:
      
      let mlTypeValue = $("#ml-type").prop("selectedIndex");
      // MachineLearning Type을 선택하지 않았을 때.
      if (mlTypeValue == 0) {
        $("#ml-type").focus();
        return false;
      }

      // 활성화된 select에 값이 있는지 체크.
      if (!CheckElementEmptyValue($("select[name=select-dbinfo]"))) {
        return false;
      }

      // 데이터 분리 input 유효성 체크.
      if (SplitValidation(null, singleton.id) == false) {
        return false;
      }

      // save model 유효성 체크
      let $saveModelCheckBox = $("#" + singleton.id["save_model_checkbox"]);
      if ($saveModelCheckBox.is(":checked") == true) {
        if (!CheckElementEmptyValue(
          $("[id*='t-save-model'][type!='checkbox']"))) {
          return false;
        }
      }

      btnTitle = (mlTypeValue == SelectedMlType.TRAIN) ? "TRAIN" : "PREDICT";
      break;

    default:
      console.log("clicked page not found.");
      break;
  }

  let parameters = {};
  let $jqueryPageId = $(jqueryPageId);
  RecursiveIterate(singleton.id, parameters, $jqueryPageId, null,
    (iterObj, targetObj, $rootJqueryObject, key, prefix) => {
      let $selector = $("#" + iterObj[key]);
      // 한 page 에 두 card의 설정 값이 있을 경우 
      // 각자 card의 설정 값만 순회하기 위해 체크.
      if (($rootJqueryObject) &&
        (!$rootJqueryObject.find($selector).length)) {
        return;
      }

      let type = $selector.attr("type");
      let value = null;

      if ($selector.prop("disabled") == false) {
        if (type === "checkbox") {
          value = $selector.is(":checked");
        } else {
          value = $selector.val();
        }
      }

      // Object 하위에 다시 Object 일 경우.
      if (prefix) {
        // 하위 Object에 키가 없으면 새로 생성.
        if (!targetObj.hasOwnProperty(prefix)) {
          targetObj[prefix] = {};
        }
        targetObj[prefix][key] = value;

      } else {
        // Object에 키가 없을 경우 새로 생성.
        if (!targetObj.hasOwnProperty(key)) {
          targetObj[key] = {};
        }
        targetObj[key] = value;
      }
    }
  );

  //console.log(parameters);
  //console.log("map length (insert before):", singleton.map.size);

  let uuid = uuidv4();
  if (uuid.length == 0) {
    console.log("uuidv4 allocate failed.");
    return false;
  }

  let btnDetail = "Task ID : " + uuid;

  // footer에 추가.
  let $listGroup = $("<a></a>")
    .addClass("list-group-item list-group-item-action d-flex")
    .addClass("justify-content-between align-items-center")
    .attr("task-id", uuid)
    .attr("id", "title-" + uuid)
    .attr("data-bs-toggle", "list")
    .attr("href", "#detail-" + uuid)
    .attr("role", "tab")
    .attr("aria-controls", "detail-" + uuid)
    .text(btnTitle);

  $listGroup.append(
    $("<button></button>").addClass("btn-close btn-sm")
      .attr("aria-label", "Close")
      .attr("name", "btn-task-close"));

  // 왼쪽 ListGroup 에 추가
  $("#task-title").append($listGroup);

  let contentString = "";
  for (const [key, value] of Object.entries(parameters)) {

    let convertValue = "";
    if (typeof (value) === "boolean") {
      convertValue = value === true ? "true" : "false";
    } else if (typeof (value) === "object") {
      convertValue = JSON.stringify(value);
    } else {
      convertValue = value != "" ? value : "null";
    }
    contentString += StringFormat("%% : %% <br/>", key, convertValue);
  }

  let $tabContents = $("<div></div>")
    .addClass("tab-pane fade")
    .attr("id", "detail-" + uuid)
    .attr("role", "tabpanel")
    .attr("aria-labelledby", "title-" + uuid)
    .html(contentString);
  //.text(JSON.stringify(parameters));

  // 오른쪽 tabContent 에 추가
  $("#task-detail").append($tabContents);

  let taskInfo = new TaskInformation();
  taskInfo.pageName = clickedPageName;
  taskInfo.btnTitle = btnTitle;
  taskInfo.btnDetail = btnDetail;
  taskInfo.parameters = parameters;

  singleton.map.set(uuid, taskInfo);

  // add badge.
  if ($("#btn-show-task-list > span").length == 0) {
    $("#btn-show-task-list").append(
      $("<span></span>").addClass("position-absolute top-0 start-100 \
      translate-middle p-1 bg-danger rounded-circle").append(
        $("<span></span>").addClass("visually-hidden"))
    );
  }

  return true;
}

function BtnExecuteClick() {

  let $taskList = $("footer a");
  if ($taskList.length == 0) {
    return true;
  }

  //console.log("map length:", singleton.map.size);
  let sendDataArray = [];
  $taskList.each(function (index, item) {

    // 이미 실행중인 task가 있으면 continue.
    if ($(item).attr("disabled")) {
      return true;
    }

    // task id가 없으면 continue.
    let taskId = $(item).attr("task-id");
    if (!taskId) {
      return true;
    }

    let jsonData = singleton.map.get(taskId).parameters;
    if (!jsonData) {
      console.log("get data empty.");
      return true;
    }

    jsonData["task_id"] = taskId;
    jsonData["page_name"] = singleton.map.get(taskId).pageName;

    sendDataArray.push(JSON.stringify(jsonData));
  });

  if (sendDataArray.length <= 0) {
    console.log("sendDataArray length <= 0. return false.");
    return false;
  }

  // footer task list 비활성 처리.
  $taskList.attr("disabled", true);
  // footer task list close 버튼 비활성 처리.
  $("footer div.card-body button.btn-close").attr("disabled", true);

  CallSyncAjaxList("task", "POST", sendDataArray,
    (json, data) => {
      // 개별 성공 callback.
      //console.log("success: ", json, data);
    },
    (json) => {
      // 실패할 경우 task list 정리.
      let $taskList = $("footer a");
      if ($taskList) {
        $taskList.each(function (index, item) {
          let taskId = $(item).attr("task-id");
          if (taskId) {
            singleton.map.delete(taskId);
            $("#title-" + taskId).remove();
            $("#detail-" + taskId).remove();
          }
        });
      }

      if (singleton.map.size == 0) {
        $("#btn-show-task-list > span").remove();
      }
    }, (json) => {
      // 전체 ajax 완료 callback.
      //console.log("done.");
    }
  );
}

$("footer").on("click", "button[name='btn-task-close']", function (event) {

  let btnTaskId = $(this).parent().attr("task-id");
  singleton.map.delete(btnTaskId);
  //console.log("delete: ", btnTaskId, " ", singleton.map.size);

  $("#title-" + btnTaskId).remove();
  $("#detail-" + btnTaskId).remove();

  // remove badge.
  if (singleton.map.size == 0) {
    $("#btn-show-task-list > span").remove();
  }
});

$(window).resize(function () {

  // width 가 991 이하로 줄어들면 navbar가 아이콘으로 바뀌면서 
  // 메뉴가 아래로 확장됨
  if ($(window).width() <= 991) {
    // navbar가 확장된 상태라면
    if ($("#navbarSupportedContent").hasClass("show") == true) {

      if ($("#btn-show-task-list").hasClass("pt-nav-collapse-show") == false) {
        // show task 버튼을 확장된 navbar 아래로 이동
        $("#btn-show-task-list").removeClass("pt-nav-collapse-hide");
        $("#btn-show-task-list").addClass("pt-nav-collapse-show");

        RemoveClassList([$("#main-row"), $("footer")], "mg-bt-64");
        RemoveClassList([$("#main-row"), $("footer")], "mg-bt-192");

        if ($("footer").hasClass("d-none")) {
          // footer 가 숨어있을 경우 main-row 를 위로 당김
          AddClass($("#main-row"), "mg-bt-192");
        } else {
          // footer 가 있을 경우 위로 당김
          AddClass($("footer"), "mg-bt-192");
        }
      }
    }
  } else {
    // width 가 991 초과하면 확장 되어 있던 navbar 메뉴도 축소됨
    if ($("#btn-show-task-list").hasClass("pt-nav-collapse-hide") == false) {
      // navbar 메뉴가 축소되면 show task 버튼도 위로 옮김
      $("#btn-show-task-list").removeClass("pt-nav-collapse-show");
      $("#btn-show-task-list").addClass("pt-nav-collapse-hide");

      RemoveClassList([$("#main-row"), $("footer")], "mg-bt-64");
      RemoveClassList([$("#main-row"), $("footer")], "mg-bt-192");

      if ($("footer").hasClass("d-none")) {
        // footer 가 숨어있을 경우 main-row 를 위로 당김
        AddClass($("#main-row"), "mg-bt-64");
      } else {
        // footer 가 있을 경우 위로 당김
        AddClass($("footer"), "mg-bt-64");
      }
    }
  }
});

$('.collapse').on('show.bs.collapse', function () {

  RemoveClass($("#btn-show-task-list"), "pt-nav-collapse-hide");
  AddClass($("#btn-show-task-list"), "pt-nav-collapse-show");

  RemoveClassList([$("#main-row"), $("footer")], "mg-bt-64");
  RemoveClassList([$("#main-row"), $("footer")], "mg-bt-192");

  if ($("footer").hasClass("d-none")) {
    // footer 가 숨어있을 경우 main-row 를 위로 당김
    AddClass($("#main-row"), "mg-bt-192");
  } else {
    // footer 가 있을 경우 위로 당김
    AddClass($("footer"), "mg-bt-192");
  }
});

$('.collapse').on('hide.bs.collapse', function () {

  RemoveClass($("#btn-show-task-list"), "pt-nav-collapse-show");
  AddClass($("#btn-show-task-list"), "pt-nav-collapse-hide");

  RemoveClassList([$("#main-row"), $("footer")], "mg-bt-64");
  RemoveClassList([$("#main-row"), $("footer")], "mg-bt-192");

  if ($("footer").hasClass("d-none")) {
    // footer 가 숨어있을 경우 main-row 를 위로 당김
    AddClass($("#main-row"), "mg-bt-64");
  } else {
    // footer 가 있을 경우 위로 당김
    AddClass($("footer"), "mg-bt-64");
  }
});

function ToggleTaskListButton() {

  RemoveClassList([$("#main-row"), $("footer")], "mg-bt-64");
  RemoveClassList([$("#main-row"), $("footer")], "mg-bt-192");

  // footer 가 hide 되어 있을 경우
  if ($("footer").hasClass("d-none") == true) {
    // footer 보이기
    $("footer").removeClass("d-none");

    if ($("#btn-show-task-list").hasClass("pt-nav-collapse-hide")) {
      // navbar가 확장 되지 않았으면 기본 navbar 만큼 당김
      $("footer").addClass("mg-bt-64");
    } else {
      // navbar가 확장 되어 있으면 확장된 navbar 만큼 당김
      $("footer").addClass("mg-bt-192");
    }
  } else {
    // footer 숨기기
    $("footer").addClass("d-none");

    if ($("#btn-show-task-list").hasClass("pt-nav-collapse-hide")) {
      // navbar가 확장 되지 않았으면 기본 navbar 만큼 당김
      $("#main-row").addClass("mg-bt-64");
    } else {
      // navbar가 확장 되어 있으면 확장된 navbar 만큼 당김
      $("#main-row").addClass("mg-bt-192");
    }
  }
}
