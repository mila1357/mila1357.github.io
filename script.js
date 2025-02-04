async function loadPyodideAndRun() {
    window.pyodide = await loadPyodide();
    await pyodide.loadPackage("numpy");
    await pyodide.loadPackage("pandas");
    await pyodide.loadPackage("scikit-learn");
    await pyodide.loadPackage("transformers");
    await pyodide.loadPackage("torch");
}
loadPyodideAndRun();

async function getRecommendations() {
    let movieInput = document.getElementById("movieInput").value;
    let movieList = movieInput.split(",").map(movie => movie.trim());
    
    let pythonCode = `
import json
user_inputs = ${JSON.stringify(movieList)}
result = recommend_movies(user_inputs, df, df_final)
json.dumps(result)
    `;
    
    let result = await pyodide.runPythonAsync(pythonCode);
    let recommendations = JSON.parse(result);
    
    displayResults(recommendations);
}

function displayResults(recommendations) {
    let movieListElem = document.getElementById("recommendedMovies");
    let newsListElem = document.getElementById("recommendedNews");
    
    movieListElem.innerHTML = "";
    newsListElem.innerHTML = "";
    
    recommendations["Recommended Movies"].forEach(movie => {
        let li = document.createElement("li");
        li.textContent = movie;
        movieListElem.appendChild(li);
    });
    
    recommendations["Recommended News"].forEach(news => {
        let li = document.createElement("li");
        let a = document.createElement("a");
        a.href = news.link;
        a.textContent = news.title;
        a.target = "_blank";
        li.appendChild(a);
        newsListElem.appendChild(li);
    });
}
