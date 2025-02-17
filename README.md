<img src="./images/landingpageimg.jpg" width="100%" height="300">

    
<script>
    fetch("./index.html")
        .then(response => response.text())
        .then(data => document.innerHTML = data);
</script>
