async function sendQuestion(){
    const input = document.getElementById("question")
    const question = input.value.trim();
    if(question==="") return;
    const chat = document.getElementById("chat-box");
    chat.innerHTML +=
        `<div class="user"><span>${question}</span></div>`;
    input.value="";

    const response = await fetch("/ask",{
        method:"POST",
        headers:{
            "Content-Type":"application/json"
        },
        body:JSON.stringify({
            question:question
        })
    });

    const data = await response.json();
    chat.innerHTML +=
        `<div class="bot"><span>${data.answer}</span></div>`;
    chat.scrollTop = chat.scrollHeight;
}