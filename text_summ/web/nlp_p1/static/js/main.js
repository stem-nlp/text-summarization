function fillOutput(result){
}

$("#start-button").click(()=>{
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
        data: {"title":title, "body":body},
        success: (response)=>{
            window.location.hash = "#start-button";
            renderOutput(response.data);
        }
    });

});

// 渲染结果
function renderOutput(result){
    $("#output").text(result.content);
    let cnt=0;
    $("#result-table tbody").empty();
    for (let line of result.detail){
        cnt += 1;
        let tableLine = `<tr>
                            <td>${cnt}</td>
                            <td>${line.content}</td>
                            <td>${line.score}</td>
                            <td>${line.reserved ? "是" : "否"}</td>
                        </tr>`;
        $("#result-table tbody").append($(tableLine))
    }
}
