/**
* Template Name: ComingSoon - v4.8.1
* Template URL: https://bootstrapmade.com/comingsoon-free-html-bootstrap-template/
* Author: BootstrapMade.com
* License: https://bootstrapmade.com/license/
*/
const form = document.getElementById('form');
const happy = document.getElementById('happy');
const good = document.getElementById('good');
const angry = document.getElementById('angry');
const prob = document.getElementById("probability");
const loader = $("div.loader");
function test() {
    _loadPage();
    var text = $('#chat').val();

    $.ajax({
        type: 'POST',
        url: 'http://127.0.0.1:5000/model',
        data: {
            chat: text
        },
        dataType: 'JSON',
        success: function (result) {
            const probability = JSON.parse(result)
            prob.innerText = probability
            if (probability > 60) {
                happy.classList.add('invisible')
                good.classList.add('invisible')
                angry.classList.remove('invisible')
            } else if (probability > 40) {
                happy.classList.add('invisible')
                good.classList.remove('invisible')
                angry.classList.add('invisible')
            } else {
                happy.classList.remove('invisible')
                good.classList.add('invisible')
                angry.classList.add('invisible')
            }
            _showPage();
        },
        error: function (xtr, status, error) {
            alert(xtr + ":" + status + ":" + error);
        }
    });
}



var _showPage = function () {
    loader.css("display", "none");
};

var _loadPage = function () {
    loader.css("display", "visible");
};