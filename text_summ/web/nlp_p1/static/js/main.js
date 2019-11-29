function fillOutput(result){
}

$("#start-button").click(()=>{
    showSummary(0.2);
});

// 渲染结果
function renderOutput(result){
    $("#output").text(result.content);
//    let cnt=0;
//    $("#result-table tbody").empty();
//    for (let line of result.detail){
//        cnt += 1;
//        let tableLine = `<tr>
//                            <td>${cnt}</td>
//                            <td>${line.content}</td>
//                            <td>${line.score}</td>
//                            <td>${line.reserved ? "是" : "否"}</td>
//                        </tr>`;
//        $("#result-table tbody").append($(tableLine))
//    }
}

// 根据百分比显示相应结果
function percentOutput(){
    var btnGroup = document.getElementById("btnGroup");
    var btns = btnGroup.getElementsByTagName("label");
    btns[3].onclick = function(){
        showSummary(0.5);
    }

    btns[2].onclick = function(){
        showSummary(0.4);
    }

    btns[1].onclick = function(){
        showSummary(0.3);
    }

    btns[0].onclick = function(){
        showSummary(0.2);
    }
}
percentOutput();

function showSummary(percent){
     let title = $("#input-title").val();
    let body = $("#input-body").val();
    let popup = $("#popup");
    if (title.length === 0){
        popup.text("请输入新闻标题").show();
        return
    }
    if (body.length === 0){
        popup.text("请输入新闻正文").show();
        return
    }
    popup.text("").hide();

    // http 请求
    $.ajax({
        url:"/api/model",
        type: "post",
        dataType: "json",
        data: {"title":title, "body":body,"percent":percent},
        success: (response)=>{
            window.location.hash = "#start-button";
            renderOutput(response.data);
        }
    });

}


