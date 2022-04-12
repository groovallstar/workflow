'use strict';

const PageReadyPromise = (sendData,
  attributesCallback,
  contentsCallback,
  getDatabaseListCallback,
  settingsCallback,
) => {
  // ajax "/attributes" 
  AttributesAjax(sendData)
    .then((result) => {
      if (attributesCallback) {
        attributesCallback(result); // callback
      }
      // ajax "/contents" 
      return ContentsAjax(sendData);
    })
    .then((result) => {
      if (contentsCallback) {
        contentsCallback(result); // callback
      }
      // ajax "/lists" 
      return GetDatabaseListAjax(sendData);
    })
    .then((result) => {
      if (getDatabaseListCallback) {
        getDatabaseListCallback(result); // callback
      }
      // ajax "/settings" 
      return SettingsAjax(sendData);
    })
    .then((result) => {
      if (settingsCallback) {
        settingsCallback(result); // callback
      }
    })
    .catch((xhr, ajaxOptions, thrownError) => {
      console.error(xhr, ajaxOptions, thrownError);
    });
};

function InitializePage(pageName) {

  if (!pageName) {
    return false;
  }

  let sendData = { "page": pageName };

  PageReadyPromise(sendData,
    (result) => {
      // attributes
      if (result.hasOwnProperty("id")) {
        singleton.id = result["id"];
      }
      if (result.hasOwnProperty("train_prefix")) {
        singleton.trainPrefix = result["train_prefix"];
      }
      if (result.hasOwnProperty("predict_prefix")) {
        singleton.predictPrefix = result["predict_prefix"];
      }
      if (result.hasOwnProperty("all_prefix")) {
        singleton.allPrefix = result["all_prefix"];
      }
      //console.log("attributes:", result["id"]);
    },
    (result) => {
      // contents

      // Load Page.
      $("#main-contents").children().remove();
      $("#main-contents").append(result);

      // Machine Learning Type Select Event listener.
      if ($("#ml-type").length) {
        $("#ml-type").on("change", function () {

          // 학습/예측/전체에 사용하는 id는 prefix가 포함되어 있음.
          // 학습: t-
          // 예측: p-
          // 둘다 사용: a-
          let trainPrefix = StringFormat("[id^=%%]", singleton.trainPrefix);
          let predictPrefix = StringFormat("[id^=%%]", singleton.predictPrefix);
          let allPrefix = StringFormat("[id^=%%]", singleton.allPrefix);

          let $disableSelector = null;

          let mlTypeValue = $("#ml-type").prop("selectedIndex");

          if (mlTypeValue == SelectedMlType.TRAIN) {
            // 학습을 선택 할 경우
            $(trainPrefix).prop("disabled", false);
            $(allPrefix).prop("disabled", false);
            $(predictPrefix).prop("disabled", true);

            $disableSelector = $(predictPrefix);

            // 모델 저장 기능은 기본 checkbox 상태에 따라 다름
            ToggleDisableElement(
              $("#" + singleton.id["save_model_checkbox"]),
              $("[id*='t-save-model'][type!='checkbox']"));

          } else if (mlTypeValue == SelectedMlType.PREDICT) {
            // 예측을 선택 할 경우
            $(allPrefix).prop("disabled", false);
            $(trainPrefix).prop("disabled", true);
            $(predictPrefix).prop("disabled", false);

            $disableSelector = $(trainPrefix);

          } else {
            // 둘다 선택하지 않을 경우
            RecursiveIterate(singleton.id, null, null, null,
              (iterObj, targetObj, $rootJqueryObject, key, prefix) => {
                let $selector = $("#" + iterObj[key]);
                // 한 page 에 두 card의 설정 값이 있을 경우 
                // 각자 card의 설정 값만 순회하기 위해 체크.
                if (($rootJqueryObject) &&
                  (!$rootJqueryObject.find($selector).length)) {
                  return;
                }
                $selector.prop("disabled", true);
              });
          }

          if ($disableSelector) {
            $disableSelector.each(function (index, item) {
              if ($(item).attr("type") === "checkbox") {
                $(item).prop("checked", false);
              } else {
                $(item).val("");
              }
              RemoveClass($(item), "is-invalid");
              RemoveErrorMessage($(item));
            });
          }

          // Train/Predict 선택 시 thresholds 값이 없을 경우는
          // 해당 input disable 처리.
          if (singleton.id.hasOwnProperty("show_metric_by_thresholds")) {
            let $showMetricByThresholdsInput = $(
              "#" + singleton.id["show_metric_by_thresholds"]);
            if (!$showMetricByThresholdsInput.val()) {
              $showMetricByThresholdsInput.prop("disabled", true);
              $showMetricByThresholdsInput.val("");
              RemoveClass($showMetricByThresholdsInput, "is-invalid");
              RemoveErrorMessage($showMetricByThresholdsInput);
            }
          }
        }); // $("#ml-type").on("change")
      }

      // threshold input enable/disable Event listener.
      if (singleton.id.hasOwnProperty("thresholds")) {
        let $thresholdsCheckbox = $("#" + singleton.id["thresholds"]);
        if ($thresholdsCheckbox.length) {
          $thresholdsCheckbox.on("change", function () {
            ToggleDisableElement(
              $thresholdsCheckbox,
              $("#" + singleton.id["show_metric_by_thresholds"]),
              "0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9");
          }); // $thresholdsCheckbox.on("change")
        }
      }

      // save model select enable/disable Event listener.
      if (singleton.id.hasOwnProperty("save_model_checkbox")) {
        let $saveModelCheckBox = $("#" + singleton.id["save_model_checkbox"]);
        if ($saveModelCheckBox.length) {
          $saveModelCheckBox.on("change", function () {
            ToggleDisableElement(
              $("#" + singleton.id["save_model_checkbox"]),
              $("[id*='t-save-model'][type!='checkbox']"));
          }); // $saveModelCheckBox.on("change")

          let $saveModelPath = $("#" + singleton.id["save_model"]["path"]);
          $saveModelPath.on("focusout", function () {
            let filePath = $(this).val();
            if ((filePath.indexOf(".") != -1) || 
                (filePath.indexOf("/") == -1)) {
              AddClass($(this), "is-invalid");
              AddErrorMessage($(this));
            } else {
              RemoveClass($(this), "is-invalid");
              RemoveErrorMessage($(this));
            }
          });
        }
      }

      // Add Custom Event Listener.
      const selectElementName = "select[name=select-dbinfo]";
      $(selectElementName).on("click", (event) => {
        SelectChangeProcedure(event, selectElementName);
      });

      // 데이터 분할 Event Listener.
      $("input[name=input-data-split]").on({
        "keypress": NumberInputKeyPressCheck,
        "focusout": (event) => {
          SplitValidation(event, singleton.id);
        }
      });

      // row 내 단일 input element에 대한 Event Listener.
      $("input[name=input-etc]").on(EtcInputProcedure);

      // threshold input FocusOut Event Listener.
      if (singleton.id.hasOwnProperty("show_metric_by_thresholds")) {
        $("#" + singleton.id["show_metric_by_thresholds"]).on(
          ThresholdFocusOutProcedure);
      }

      // Insert Data 페이지 날짜 input validate check.
      if (singleton.id.hasOwnProperty("date")) {
        let $date = $("#" + singleton.id["date"]);
        if ($date.length) {
          $date.on("focusout", function () {
            let value = $(this).val().trim();
            if (value === "") {
              RemoveClass($(this), "is-invalid");
              return true;
            }

            // 정규표현식에 맞지 않을 경우
            const pattern = /(^(20)\d{2})(0[1-9]|1[0-2])$/;
            if (!pattern.test(value)) {
              AddClass($(this), "is-invalid");
              // div의 invalid-feedback가 추가되어 있으면
              // class에 is-invalid 추가 시 해당 input 아래에 보이게 됨
              let errorMessage = $(this).nextAll("label").attr("error-message");
              if (errorMessage.length > 0) {
                $(this).siblings(".invalid-feedback").text(errorMessage);
              }
            } else {
              RemoveClass($(this), "is-invalid");
            }
          });
        }
      }
    },
    (result) => {
      // lists.
      let $dbSelect = $("select[name=select-dbinfo][id$='database']");
      if (($dbSelect.length > 0) && (result.hasOwnProperty("database"))) {
        // set select database list.
        $.each(result['database'], function (key, value) {
          $dbSelect.append($("<option></option>").text(value));
        });
      }
    },
    (result) => {
      // settings.
      let $jqueryPageId = $(StringFormat("[id='%%']", pageName));
      // 학습/예측 페이지일 경우.
      if ($("#ml-type").length) {

        $("#ml-type").prop("selectedIndex", 0);

        if (result) {
          RecursiveIterate(singleton.id, result, $jqueryPageId, null,
            (iterObj, targetObj, $rootJqueryObject, key, prefix) => {
              let $selector = $("#" + iterObj[key]);
              // 한 page 에 두 card의 설정 값이 있을 경우 
              // 각자 card의 설정 값만 순회하기 위해 체크.
              if (($rootJqueryObject) &&
                (!$rootJqueryObject.find($selector).length)) {
                return;
              }

              let value = null;
              if (prefix) {
                if (targetObj.hasOwnProperty(prefix)) {
                  value = targetObj[prefix][key];
                }
              } else {
                value = targetObj[key];
              }

              if (value) {
                if (typeof (value) === "boolean") {
                  if (value == true) {
                    $selector.attr("checked", true);
                  } else {
                    $selector.removeAttr("checked");
                  }
                } else {
                  if ($selector.is("select")) {
                    // select list 에 값이 없을 경우에만 추가함.
                    if (AlreadyExistSelectValue($selector, value) == false) {
                      $selector.append(
                        $("<option></option>").text(value)
                          .attr("selected", "selected"));
                    } else {
                      $selector.val(value);
                    }
                  } else {
                    $selector.val(value);
                  }
                }
                $selector.prop("disabled", true);
              } else {
                $selector.prop("disabled", true);
              }
            });

        } else {
          // 마지막 설정 값이 없으면 모든 Element 비활성화.
          RecursiveIterate(singleton.id, result, $jqueryPageId, null,
            (iterObj, targetObj, $rootJqueryObject, key, prefix) => {
              let $selector = $("#" + iterObj[key]);
              // 한 page 에 두 card의 설정 값이 있을 경우 
              // 각자 card의 설정 값만 순회하기 위해 체크.
              if (($rootJqueryObject) &&
                (!$rootJqueryObject.find($selector).length)) {
                return;
              }
              $selector.prop("disabled", true);
            });
        }

      } else {
        // 나머지 페이지일 경우.
        if (!result) {
          console.log("last setting data empty.");
          return false;
        }

        RecursiveIterate(singleton.id, result, $jqueryPageId, null,
          (iterObj, targetObj, $rootJqueryObject, key, prefix) => {
            let $selector = $("#" + iterObj[key]);
            // 한 page 에 두 card의 설정 값이 있을 경우 
            // 각자 card의 설정 값만 순회하기 위해 체크.
            if (($rootJqueryObject) &&
              (!$rootJqueryObject.find($selector).length)) {
              return;
            }

            let value = null;
            if (prefix) {
              if (targetObj.hasOwnProperty(prefix)) {
                value = targetObj[prefix][key];
              }
            } else {
              value = targetObj[key];
            }

            if (value) {
              if (typeof (value) === "boolean") {
                if (value == true) {
                  $selector.attr("checked", true);
                } else {
                  $selector.removeAttr("checked");
                }
              } else {
                if ($selector.is("select")) {
                  // select list 에 값이 없을 경우에만 추가함.
                  if (AlreadyExistSelectValue($selector, value) == false) {
                    $selector.append(
                      $("<option></option>").text(value)
                        .attr("selected", "selected"));
                  } else {
                    $selector.val(value);
                  }
                } else {
                  $selector.val(value);
                }
              }
            }
          });
      }
    });
}
