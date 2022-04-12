'use strict';

function SelectChangeProcedure(event, elementName) {

  if ((!event) || (!elementName)) {
    return false;
  }

  const $this = $(event.currentTarget);

  // column
  let $column = $this.parent().parent().parent().children().find(elementName);
  if ($column.length <= 0) {
    return false;
  }

  // database
  let $dbSelect = $column.eq(0);
  let databaseValue = $dbSelect.val();

  // collection
  let $colSelect = $column.eq(1);
  let columnValue = $colSelect.val();

  // startdate
  let $startDateSelect = $column.eq(2);
  let startDateValue = $startDateSelect.val();

  // enddate
  let $endDateSelect = $column.eq(3);
  let endDateValue = $endDateSelect.val();

  let sendData = {};
  if (databaseValue) sendData['database'] = databaseValue;
  if (columnValue) sendData['collection'] = columnValue;
  if (startDateValue) sendData['start_date'] = startDateValue;
  if (endDateValue) sendData['end_date'] = endDateValue;

  // 각 select에 값이 없을 경우
  if (Object.keys(sendData).length == 0) {

    // 각 select 초기화
    InitSelectList([$colSelect, $startDateSelect, $endDateSelect]);
    RemoveClassList(
      [$dbSelect, $colSelect, $startDateSelect, $endDateSelect],
      "is-invalid");
    RemoveErrorMessage($this);
    return false;
  }

  // 2개 또는 4개 컬럼에 모두 값이 있을 때는 개수 체크함.
  if (Object.keys(sendData).length === $column.length) {

    CallAjax("count", "GET", sendData,
      (responseData) => {
        let isValid = true;
        if (!responseData) {
          isValid = false;

        } else {
          if (responseData.hasOwnProperty("count")) {
            //console.log("count:(",responseData["count"],")");
            if (responseData["count"] > 0) {
              isValid = true;
            } else {
              isValid = false;
            }
          }
        }

        // 유효성 체크 성공.
        if (isValid == true) {
          RemoveClassList(
            [$dbSelect, $colSelect, $startDateSelect, $endDateSelect],
            "is-invalid");
          RemoveErrorMessage($this);

        } else {
          // 유효성 실패. 
          InitSelectList([$colSelect, $startDateSelect, $endDateSelect]);
          AddClassList(
            [$dbSelect, $colSelect, $startDateSelect, $endDateSelect],
            "is-invalid");
          AddErrorMessage($this);
        }
      }); // CallAjax.

    return true;

  } else {

    // collection 명을 제한하기 위해 추가로 id 전달.
    sendData["id"] = $this.attr("id");
    CallAjax("lists", "GET", sendData,
      (responseData) => {

        let isValid = true;
        if (!responseData) {
          isValid = false;

        } else {
          if (responseData.hasOwnProperty("collection")) {
            // 컬렉션 목록
            if (responseData["collection"].length === 0) {
              isValid = false;
            } else {
              if ($colSelect.length > 0) {
                $colSelect.html($("<option selected></option>"));
                $.each(responseData['collection'], function (key, value) {
                  $colSelect.append($("<option></option>").text(value));
                });
              }
              InitSelectList([$startDateSelect, $endDateSelect]);
            }
          } else if (responseData.hasOwnProperty("start_date")) {
            // 시작날짜
            if (responseData["start_date"].length === 0) {
              isValid = false;
            } else {
              if ($startDateSelect.length > 0) {
                $startDateSelect.html($("<option selected></option>"));
                $.each(responseData['start_date'], function (key, value) {
                  $startDateSelect.append(
                    $("<option></option>").text(new Date(value).yyyymm()));
                });
                InitSelectList([$colSelect, $endDateSelect]);
              }
            }
          } else if (responseData.hasOwnProperty("end_date")) {
            // 종료날짜
            if (responseData["end_date"].length === 0) {
              isValid = false;
            } else {
              if ($endDateSelect.length > 0) {
                $endDateSelect.html($("<option selected></option>"));
                $.each(responseData['end_date'], function (key, value) {
                  $endDateSelect.append(
                    $("<option></option>").text(new Date(value).yyyymm()));
                });
                InitSelectList([$colSelect, $startDateSelect]);
              }
            }
          }
        }

        // 유효성 체크 성공.
        if (isValid == true) {
          RemoveClassList(
            [$dbSelect, $colSelect, $startDateSelect, $endDateSelect],
            "is-invalid");
          RemoveErrorMessage($this);

        } else {
          // 유효성 실패.
          AddClassList(
            [$dbSelect, $colSelect, $startDateSelect, $endDateSelect],
            "is-invalid");
          AddErrorMessage($this);

          InitSelectList([$colSelect, $startDateSelect, $endDateSelect]);
        }
      }); // lists CallAjax.
  }
}

const ThresholdFocusOutProcedure = {
  "keypress": function (event) {
    if (event.which < 48 || event.which > 57) {
      if ((event.which != 46) && (event.which != 44) && (event.which != 32)) {
        // Allow '.', ',', ' ', number.
        event.preventDefault();
        return true;
      }
    }
  },
  "focusout": function () {
    const pattern = /^(\s*-?\d+(\.\d+)?)(\s*,\s*-?\d+(\.\d+)?)*$/;
    //console.log("Regular expression result: ", pattern.test($(this).val()));
    let value = $(this).val().trim();
    // 정규표현식에 맞지 않을 경우
    if (!pattern.test(value)) {
      AddClass($(this), "is-invalid");
      AddErrorMessage($(this));
      return;
    }

    let isValid = true;
    try {
      value.split(',').forEach(element => {
        let convertElement = parseFloat(element.trim());
        if ((isNaN(convertElement)) || (typeof convertElement !== "number")) {
          isValid = false;
          return false;
        }

        if ((convertElement <= 0) || (convertElement > 1.0)) {
          isValid = false;
          return false;
        }
      });
    } catch (err) {
      console.log(err);
      isValid = false;
    }

    if (!isValid) {
      AddClass($(this), "is-invalid");
      AddErrorMessage($(this));
      return false;
    }
    RemoveClass($(this), "is-invalid");
    RemoveErrorMessage($(this));
    return true;
  }
};

const EtcInputProcedure = {
  "keypress": NumberInputKeyPressCheck,
  "focusout": function (event) {
    let thisValue = parseFloat($(this).val());
    let maxValue = parseFloat($(this).prop("max"));
    // 1.0 이하 값은 소수점 처리.
    if (maxValue <= 1.0) {
      maxValue = Number(maxValue).toFixed(1);
    }
    if (thisValue > maxValue) {
      AddClass($(this), "is-invalid");
      // div의 invalid-feedback가 추가되어 있으면
      // class에 is-invalid 추가 시 해당 input 아래에 보이게 됨
      let errorMessage = $(this).nextAll("label").attr("error-message");
      if (errorMessage.length > 0) {
        $(this).siblings(".invalid-feedback").text(
          maxValue + ' ' + errorMessage);
      }
    } else {
      RemoveClass($(this), "is-invalid");
    }
  }
};

function SplitValidation(event, id) {

  if (!id.hasOwnProperty("split_ratio")) {
    AddClass($testInput, "is-invalid");
    AddErrorMessage($thisSelector);
    return false;
  }

  let mlTypeValue = $("#ml-type").prop("selectedIndex");
  let $trainInput = $("#" + id["split_ratio"]["train"]);
  let $validationInput = $("#" + id["split_ratio"]["validation"]);
  let $testInput = $("#" + id["split_ratio"]["test"]);

  if ((!$trainInput.length) ||
    (!$validationInput.length) ||
    (!$testInput.length)) {
    AddClass($testInput, "is-invalid");
    AddErrorMessage($thisSelector);
    return false;
  }

  let $thisSelector = null;
  if (mlTypeValue == SelectedMlType.PREDICT) {
    $thisSelector = $testInput;
  } else {
    $thisSelector = $trainInput;
  }

  if (($trainInput.val() == "") &&
    ($validationInput.val() == "") &&
    ($testInput.val() == "")) {

    if (mlTypeValue == SelectedMlType.PREDICT) {
      AddClass($testInput, "is-invalid");
      AddErrorMessage($thisSelector);
      return false;
    }
    RemoveClassList(
      [$trainInput, $validationInput, $testInput],
      "is-invalid");
    RemoveErrorMessage($thisSelector);
    return true;
  }

  let trainValue = 0.0;
  let validationValue = 0.0;
  let testValue = 0.0;

  try {
    if ($trainInput.val().length > 0) {
      trainValue = parseFloat($trainInput.val());
    }
    if ($validationInput.val().length > 0) {
      validationValue = parseFloat($validationInput.val());
    }
    if ($testInput.val().length > 0) {
      testValue = parseFloat($testInput.val());
    }

    let total = 0.0;
    total = appendValue(total, trainValue);
    total = appendValue(total, testValue);
    if (validationValue > 0) {
      total = appendValue(total, validationValue);
    }

    if ((total > 0) && (total != 1.0)) {

      if (mlTypeValue == SelectedMlType.TRAIN) {
        AddClassList(
          [$trainInput, $validationInput, $testInput],
          "is-invalid");
        AddErrorMessage($thisSelector);

      } else {
        AddClass($testInput, "is-invalid");
        AddErrorMessage($thisSelector);
      }
      return false;
    } else {

      // 총 합이 1.0 이어도 에러 처리하는 경우.
      // 1. train 값이 0 이고, validation 값이 1.0 일 경우
      // 2. train 값이 0 이고, validation+test 값이 1.0 일 경우
      if (mlTypeValue == 1) {
        if ((trainValue == 0) && (validationValue > 0)) {
          AddClassList(
            [$trainInput, $validationInput, $testInput],
            "is-invalid");
          AddErrorMessage($thisSelector);
          return false;
        }
      }

      // 유효성 체크 통과.
      RemoveClassList(
        [$trainInput, $validationInput, $testInput],
        "is-invalid");
      RemoveErrorMessage($thisSelector);
    }
    return true;

  } catch (err) {
    console.log(err);
  }
}
