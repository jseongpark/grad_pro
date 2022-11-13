/**
* Template Name: ComingSoon - v4.8.1
* Template URL: https://bootstrapmade.com/comingsoon-free-html-bootstrap-template/
* Author: BootstrapMade.com
* License: https://bootstrapmade.com/license/
*/
function submit(event) {
  good.classList.add("invisible");
  //XMLHttpRequest 객체 생성
  var xhr = new XMLHttpRequest();

  //요청을 보낼 방식, 주소, 비동기여부 설정
  xhr.open('POST', './model', true);
  //요청 전송
  xhr.send();
  //통신후 작업
  xhr.onload = () => {
      //통신 성공
      if (xhr.status == 200) {
          console.log(xhr.response);
          console.log("통신 성공");
      } else {
          //통신 실패
          console.log("통신 실패");
      }
  }
  event.preventDefault();
}

const form = document.getElementById('form');
const happy = document.getElementById('happy');
const good = document.getElementById('good');
const angry = document.getElementById('angry');
form.addEventListener('submit', submit);