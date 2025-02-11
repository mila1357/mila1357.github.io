document.addEventListener("DOMContentLoaded", function() {
    const form = document.querySelector("form");
    const resultDiv = document.querySelector("#results");

    form.addEventListener("submit", function(event) {
        event.preventDefault();  // Prevent page reload

        let userInput = document.querySelector("input[name='keyword']").value;
        resultDiv.innerHTML = "<p>Loading recommendations...</p>";

        fetch("/recommend", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ movies: [userInput] })
        })
        .then(response => response.json())
        .then(data => {
            resultDiv.innerHTML = "";  // Clear previous results

            if (data["Recommended Movies"].length === 0) {
                resultDiv.innerHTML = "<p>No recommendations found.</p>";
                return;
            }

            // Display Recommended Movies
            let movieSection = "<h2>Recommended Movies:</h2><ul>";
            data["Recommended Movies"].forEach((movie, index) => {
                movieSection += `<li><strong>${movie}</strong>: ${data["Recommended Movies Descriptions"][index]}</li>`;
            });
            movieSection += "</ul>";

            // Display Recommended News
            let newsSection = "<h2>Recommended News:</h2><ul>";
            if (data["Recommended Movies Genres"]) {
                data["Recommended Movies Genres"].forEach((genre, index) => {
                    newsSection += `<li><strong>${genre}</strong></li>`;
                });
            }
            newsSection += "</ul>";

            resultDiv.innerHTML = movieSection + newsSection;
        })
        .catch(error => {
            console.error("Error:", error);
            resultDiv.innerHTML = "<p>Failed to get recommendations.</p>";
        });
    });
});
