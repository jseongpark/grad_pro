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

function test(){
            var text = $('#chat').val();
            $.ajax({
                    type : 'POST',                                 
                    url : 'http://127.0.0.1:5000/model',
                    data : {
                           chat:text                       
                    },
                    dataType : 'JSON',
                    success : function(result){
                            alert("result = "+ result);
                    },
                    error : function(xtr,status,error){
                            alert(xtr +":"+status+":"+error);
                    }
            });
    }