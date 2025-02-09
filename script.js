function getRecommendations() {
    let inputText = document.getElementById("userInput").value;
    let userInputs = inputText.split(',').map(item => item.trim()); // Convert input into an array

    if (userInputs.length === 0 || userInputs[0] === "") {
        alert("Please enter at least one movie title.");
        return;
    }

    fetch("https://News Recommendation.netlify.app/recommend", { // Replace with actual Netlify function URL
        method: "POST",
        headers: {
            "Content-Type": "application/json"
        },
        body: JSON.stringify({ movies: userInputs })
    })
    .then(response => response.json())
    .then(data => {
        let resultDiv = document.getElementById("result");
        resultDiv.innerHTML = "<h2>Recommended Movies</h2>";
        
        if (data["Recommended Movies"].length === 0) {
            resultDiv.innerHTML += "<p>No recommendations found.</p>";
            return;
        }

        let list = "<ul>";
        data["Recommended Movies"].forEach((movie, index) => {
            list += `<li><strong>${movie}</strong> - ${data["Recommended Movies Descriptions"][index]}</li>`;
        });
        list += "</ul>";

        resultDiv.innerHTML += list;
    })
    .catch(error => console.error("Error:", error));
}
