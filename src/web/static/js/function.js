'use strict';

/**
 * 각 페이지 이름 Define.
 */
let PageName = Object.freeze({
  "InsertData": "INSERT DATA",
  "InsertTable": "INSERT TABLE",
  "TrainPredict": "TRAIN PREDICT",
});

/**
 * 학습/예측 페이지에서 타입 선택 값.
 */
let SelectedMlType = Object.freeze({
  "TRAIN": 1,
  "PREDICT": 2,
});

/**
 * 하단 footer 에서 실행할 task 에 대한 저장 데이터.
 */
class TaskInformation {
  constructor() {
    this.pageName = new String();
    this.btnTitle = new String();
    this.btnDetail = new String();
    this.parameters = new String();
  }
}

let instance = null;
class Singleton {
  constructor() {
    if (instance) {
      return instance;
    }
    this.map = new Map();

    this.id = new String();
    this.trainPrefix = new String();
    this.predictPrefix = new String();
    this.allPrefix = new String();

    instance = this;
  }
}

let singleton = new Singleton();

/**
 * 문자열 -> YYYYMM 으로 변환.
 */
Date.prototype.yyyymm = function () {
  let mm = this.getMonth() + 1; // getMonth() is zero-based
  let date = [this.getFullYear(), (mm > 9 ? '' : '0') + mm].join('');
  return date;
};

/**
 * "%%" 을 대입 문자열로 치환.
 */
function StringFormat(fmt, ...args) {
  return fmt
    .split("%%")
    .reduce((aggregate, chunk, i) => aggregate + chunk + (args[i] || ""), "");
}

/**
 * 누적합.
 *
 * @param total 누적할 변수.
 * @param value total에 더하는 값.
 */
const appendValue = (total, value) => {
  if ((isNaN(value) == false) && (isFinite(value))) {
    return Number((total += value).toFixed(2));
  }
  return Number(total.toFixed(2));
}

/**
 * ajax 호출.
 *
 * @param url 호출 URI.
 * @param type Method 타입.
 * @param sendData Web Server로 전달할 데이터.
 * @param successFunc ajax 성공 시 콜백 함수.
 * @param failFunc ajax 실패 시 콜백 함수.
 * @param alwaysFunc ajax 호출 시 항상 호출 되는 콜백 함수.
 */
const CallAjax = (url, type, sendData,
  successFunc = null,
  failFunc = null,
  alwaysFunc = null) => {

  $.ajax({
    url: url,
    type: type,
    dataType: "json",
    contentType: "application/json",
    data: sendData,

  }).done((json) => {
    if (successFunc) {
      successFunc(json);
    } else {
      //
    }
  }).fail((xhr, status, errorThrown) => {
    if (failFunc) {
      failFunc(xhr, status, errorThrown);
    } else {
      console.error("Sorry. problem", status);
    }
  }).always((xhr, status) => {
    if (alwaysFunc) {
      alwaysFunc(xhr, status);
    } else {
      //
    }
  });
};

/**
 * /contents ajax 호출.
 * 
 * 페이지 로드.
 * 
 * @param sendData Web Server로 전달할 데이터.
 */
const ContentsAjax = (sendData) => {
  return new Promise((resolve, reject) => {
    CallAjax("contents", "GET", sendData,
      (json) => {
        if (!json.hasOwnProperty("html")) {
          console.error("response data not in html data.");
          reject("response data not in html data.");
        }
        resolve(json["html"]);
      },
      (xhr, ajaxOptions, thrownError) => {
        reject(xhr, ajaxOptions, thrownError);
      });
  });
};

/**
 * /attributes ajax 호출.
 * 
 * 페이지의 id 받기.
 * 
 * @param sendData Web Server로 전달할 데이터.
 */
const AttributesAjax = (sendData) => {
  return new Promise((resolve, reject) => {
    CallAjax("attributes", "GET", sendData,
      (json) => {
        if (!json.hasOwnProperty("id")) {
          console.error("response data not in id data.");
          reject("response data not in id data.");
        }
        resolve(json);
      },
      (xhr, ajaxOptions, thrownError) => {
        reject(xhr, ajaxOptions, thrownError);
      });
  });
};

/**
 * /lists ajax 호출.
 * 
 * Database List 받기.
 * 
 */
const GetDatabaseListAjax = () => {
  return new Promise((resolve, reject) => {
    CallAjax("lists", "GET", null,
      (json) => {
        if (!json) {
          console.error("response data empty.");
          reject("response data empty.");
        }
        resolve(json);
      },
      (xhr, ajaxOptions, thrownError) => {
        reject(xhr, ajaxOptions, thrownError);
      });
  });
};

/**
 * /settings ajax 호출.
 * 
 * 페이지의 마지막 설정 값 받기.
 * 
 * @param sendData Web Server로 전달할 데이터.
 */
const SettingsAjax = (sendData) => {
  return new Promise((resolve, reject) => {
    CallAjax("settings", "GET", sendData,
      (json) => {
        if (!json) {
          console.error("response data empty.");
          resolve(); // 마지막 설정 값이 없어도 reject가 아님.
        }
        resolve(json);
      },
      (xhr, ajaxOptions, thrownError) => {
        reject(xhr, ajaxOptions, thrownError);
      });
  });
};

/**
 * 순차적으로 ajax 호출하는 함수.
 * 
 * @param url 호출할 URI.
 * @param type Method 타입.
 * @param sendDataArray 순차적으로 전달할 데이터 리스트.
 * @param elementSuccessFunc 각 ajax가 성공했을 때 콜백 함수.
 * @param elementFailFunc 각 ajax가 실패했을 때 콜백 함수.
 * @param completeFunc 모든 ajax가 완료됐을 때 콜백 함수.
 * @param exceptFunc 한 ajax라도 예외 발생했을 때의 콜백 함수.
 */
function CallSyncAjaxList(url, type, sendDataArray,
  elementSuccessFunc = null, elementFailFunc = null,
  completeFunc = null, exceptFunc = null) {
  // sendDataArray의 개별 data를 꺼내어 순차적으로 ajax 호출.
  let ajaxRequest = function (sendData) {
    let deferred = $.Deferred();
    CallAjax(url, type, sendData,
      (json) => {
        // 단일 ajax 성공.
        deferred.resolve(json);

        if (elementSuccessFunc) {
          elementSuccessFunc(json);
        }
      },
      (json) => {
        // 단일 ajax 실패.
        deferred.reject(json);
        if (elementFailFunc) {
          elementFailFunc(json);
        }
      });

    return deferred.promise();
  };

  // go through each item and call the ajax function
  let deferredResolve = $.Deferred().resolve();
  $.when.apply($, $.map(sendDataArray, function (item, i) {
    deferredResolve = deferredResolve.then(function () {
      // call ajax
      return ajaxRequest(item);
    });
    return deferredResolve;
  }))
    .then(function () {
      // array의 모든 ajax 완료 후 호출.
      if (completeFunc) {
        completeFunc();
      }
    })
    .catch(function (error) {
      if (exceptFunc) {
        exceptFunc(error);
      } else {
        console.log("Exception:", error);
      }
    });
}

/**
 * Object 객체를 depth 하위까지 순회.
 *
 * @param iterableObject 순회 할 Object.
 * @param targetObject Object 순회한 key 이름을 통해 콜백 함수 내에서 따로 접근할 Object.
 * @param rootJqueryObject 해당 JQuery Object 하위에 대상 Object 가 존재 하는지 체크.
 * @param prefix 중첩 Object 일 경우 하위 Object의 key 이름.
 * @param callback 호출할 콜백함수.
 */
const RecursiveIterate = (iterableObject, targetObject, rootJqueryObject,
  prefix, callback) => {
  Object.keys(iterableObject).forEach(key => {
    if (typeof iterableObject[key] === "object") {
      //recursive.
      RecursiveIterate(iterableObject[key], targetObject, rootJqueryObject,
        key, callback);
    } else {
      callback(iterableObject, targetObject, rootJqueryObject, key, prefix);
    }
  })
};

/**
 * 숫자 Input 에 키보드 입력에 대한 처리.
 * 
 * Input element의 최대값이 1 이상 일 경우는 10자로 제한.
 * 
 * Input element의 최대값이 1 미만이라면 3자로 제한. (e.g. 0.02).
 * 
 * @param event 키 입력 event.
 */
function NumberInputKeyPressCheck(event) {
  if (event.which < 48 || event.which > 57) {
    if ((event.which != 46)) {
      // Allow '.', number.
      //console.log(event.which);
      event.preventDefault();
      return false;
    }
  }
  if ($(this).prop("max").length > 0) {
    let thisLength = $(this).val().length;
    let maxLength = $(this).prop("max");
    // seed 설정 input.
    if (maxLength > 1) {
      if (thisLength >= 10) {
        return false;
      }
    } else {
      // float 타입 input.
      if (thisLength > 3) {
        return false;
      }
    }
  }
}

/**
 * 사용자 입력 element 하단에 에러 메시지 label 추가.
 * 
 * @param $thisSelector $(this) Jquery 셀렉터.
 */
const AddErrorMessage = $thisSelector => {
  if (($thisSelector.length <= 0)) {
    return false;
  }

  let $column = $thisSelector.parent().parent().parent().children(); // col
  if ($column.length <= 0) {
    return false;
  }

  if (!$column.last().hasClass("error-message")) {
    // 각 label에 error message attribute가 있음
    let errorMessage = $thisSelector.nextAll("label").attr("error-message");
    if (errorMessage.length > 0) {
      $column.parent().append(
        $("<label></label>").addClass("error-message").
          text(errorMessage)
      );
    }
  }
};

/**
 * 사용자 입력 element 하단에 에러 메시지 label 제거.
 * 
 * @param $thisSelector $(this) Jquery 셀렉터.
 */
const RemoveErrorMessage = $thisSelector => {
  let $column = $thisSelector.parent().parent().parent().children(); // col
  // column의 마지막 element의 error-message attribute 제거
  if ($column.length > 0) {
    if ($column.last().hasClass("error-message")) {
      $column.last().remove();
    }
  }
};

/**
 * HTML class에 지정된 class명 추가.
 * 
 * @param selectorList 적용할 Jquery 셀렉터 리스트.
 * @param className 추가할 HTML class.
 */
const AddClassList = (selectorList, className) => {
  if ((!selectorList) || (!className)) {
    return false;
  }

  if (!Array.isArray(selectorList)) {
    return false;
  }

  try {
    selectorList.forEach(function (item) {
      AddClass(item, className);
    });

  } catch (error) {
    console.error(error);
    return false;
  }

  return true;
}

/**
 * HTML class에 지정된 class명 추가.
 * 
 * @param selector 적용할 Jquery 셀렉터.
 * @param className 추가할 HTML class.
 */
const AddClass = (selector, className) => {
  if (!selector) {
    return false;
  }
  if (selector.length <= 0) {
    return false;
  }
  if (selector.hasClass(className)) {
    return false;
  }
  selector.addClass(className);
  return true;
};

/**
 * HTML class에 지정된 class명 제거.
 * 
 * @param selectorList 적용할 Jquery 셀렉터 리스트.
 * @param className 제거할 HTML class.
 */
const RemoveClassList = (selectorList, className) => {
  if ((!selectorList) || (!className)) {
    return false;
  }

  if (!Array.isArray(selectorList)) {
    return false;
  }

  try {
    selectorList.forEach(function (item) {
      RemoveClass(item, className);
    });

  } catch (error) {
    console.error(error);
    return false;
  }

  return true;
}

/**
 * HTML class에 지정된 class명 제거.
 * 
 * @param selector 적용할 Jquery 셀렉터.
 * @param className 제거할 HTML class.
 */
const RemoveClass = (selector, className) => {
  if (!selector) {
    return false;
  }
  if (selector.length <= 0) {
    return false;
  }
  if (!selector.hasClass(className)) {
    return false;
  }
  selector.removeClass(className);
  return true;
};

/**
 * HTML Select 목록 초기화.
 * 
 * @param selectorList 적용할 Select Jquery 셀렉터.
 */
const InitSelectList = (selectorList) => {
  if (!selectorList) {
    return false;
  }

  if (!Array.isArray(selectorList)) {
    return false;
  }

  selectorList.forEach(function ($selector) {
    let value = $selector.val();
    $selector.html($("<option></option>"));
    if (value) {
      $selector.append($("<option selected></option>").text(value));
    }
  });

  return true;
};

/**
 * checkbox 선택 시 하위 element 활성/비활성 토글.
 * 
 * @param $checkBox HTML checkbox Jquery 셀렉터.
 * @param $disableElement 활성/비활성 할 HTML element 셀렉터.
 * @param defaultValue 비활성 -> 활성화 될 때 기본값.
 */
const ToggleDisableElement = (
  $checkBox, $disableElement, defaultValue = "") => {

  if ($checkBox.length <= 0) {
    return false;
  }

  if ($checkBox.attr("type") != "checkbox") {
    return false;
  }

  $disableElement.each(function (index, item) {
    if ($checkBox.is(":checked")) {
      $(item).prop("disabled", false);
      if (defaultValue) {
        $(item).prop("value", defaultValue);
      }
    } else {
      $(item).prop("disabled", true);
      $(item).val("");
      RemoveClass($(item), "is-invalid");
      RemoveErrorMessage($(item));
    }
  });
};

/**
 * 해당 element 의 값이 있는지 체크.
 * 
 * @param $selector 체크할 element Jquery 셀렉터.
 * @param isFocusing 값이 없을때 해당 element에 커서 이동 여부.
 */
function CheckElementEmptyValue($selector, isFocusing = true) {

  if (!$selector) {
    return false;
  }

  if (!$selector.length) {
    return false;
  }

  let isValid = true;

  $selector.each(function (index, item) {
    if (!$(item)) {
      return true;
    }

    if ($(item).prop("disabled") == false) {
      if (!$(item).val()) {
        isValid = false;
        if (isFocusing) {
          $(item).focus();
        }
        return false;
      }
    }
  });

  return isValid;
}

/**
 * select element list 에 동일한 값이 있는지 체크.
 * 
 * @param $selector 체크할 element Jquery 셀렉터.
 * @param compareValue list에 값이 있는지 비교 값.
 */
function AlreadyExistSelectValue($selector, compareValue) {

  if (!$selector || !compareValue) {
    return false;
  }

  if (!$selector.length) {
    return false;
  }

  let $options = $selector.children();
  if (!$options.is("option")) {
    return false;
  }

  if (!$options.length) {
    return false;
  }

  let exist = false;

  $options.each(function (index, item) {
    if (!$(item)) {
      return true;
    }

    if (item.text === compareValue) {
      exist = true;
      return false;
    }
  });

  return exist;
}
