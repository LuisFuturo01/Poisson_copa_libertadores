document.addEventListener('DOMContentLoaded', () => {
    const inputLocal = document.getElementById("inputLocal");
    const listLocal = document.getElementById("listLocal");
    const wrapLocal = document.getElementById("wrapLocal");

    const inputVisitante = document.getElementById("inputVisitante");
    const listVisitante = document.getElementById("listVisitante");
    const wrapVisitante = document.getElementById("wrapVisitante");

    const btn = document.getElementById("btnPredecir");

    const imgLocal = document.getElementById("imgLocal");
    const spinnerLocal = document.getElementById("spinnerLocal");
    const imgVisitante = document.getElementById("imgVisitante");
    const spinnerVisitante = document.getElementById("spinnerVisitante");

    const ui = {
        loading: document.getElementById("loading"),
        resultado: document.getElementById("resultado"),
        localName: document.getElementById("resLocalName"),
        visitanteName: document.getElementById("resVisitanteName"),
        localVal: document.getElementById("probLocalVal"),
        empateVal: document.getElementById("probEmpateVal"),
        visitanteVal: document.getElementById("probVisitanteVal"),
        barLocal: document.getElementById("barLocal"),
        barEmpate: document.getElementById("barEmpate"),
        barVisitante: document.getElementById("barVisitante"),
    };

    const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));

    async function fetchLogoWithDelay(teamName, imgElement, spinnerElement) {
        if (!teamName) return;
        const fallback = '/static/img/shield.png';
        
        imgElement.classList.add('hidden');
        spinnerElement.classList.remove('hidden');

        const wikiPromise = (async () => {
            try {
                const searchTerm = encodeURIComponent(teamName + " club de fútbol");
                const url = `https://es.wikipedia.org/w/api.php?action=query&generator=search&gsrsearch=${searchTerm}&gsrlimit=1&prop=pageimages&pithumbsize=500&format=json&origin=*`;
                const res = await fetch(url);
                const data = await res.json();
                const pages = data.query?.pages;
                if (pages) {
                    const pageId = Object.keys(pages)[0];
                    const thumb = pages[pageId].thumbnail;
                    if (thumb && thumb.source) return thumb.source;
                }
                return null;
            } catch (e) { return null; }
        })();

        const [wikiResult] = await Promise.all([wikiPromise, wait(2000)]);

        if (wikiResult) imgElement.src = wikiResult;
        else imgElement.src = fallback;

        imgElement.onerror = function() { this.onerror = null; this.src = fallback; };
        spinnerElement.classList.add('hidden');
        imgElement.classList.remove('hidden');
    }

    function setupCustomSelect(input, list, wrapper, allTeams, onSelect) {
        
        const renderList = (filter = "") => {
            list.innerHTML = "";
            const filtered = allTeams.filter(t => t.toLowerCase().includes(filter.toLowerCase()));
            
            if (filtered.length === 0) {
                const li = document.createElement("li");
                li.textContent = "No se encontraron resultados";
                li.style.pointerEvents = "none";
                li.style.color = "#555";
                list.appendChild(li);
            } else {
                filtered.forEach(team => {
                    const li = document.createElement("li");
                    li.textContent = team;
                    li.addEventListener("click", () => {
                        input.value = team;
                        list.classList.add("hidden");
                        wrapper.classList.remove("active");
                        onSelect(team);
                    });
                    list.appendChild(li);
                });
            }
        };

        input.addEventListener("focus", () => {
            renderList(input.value); 
            list.classList.remove("hidden");
            wrapper.classList.add("active");
        });

        input.addEventListener("input", (e) => {
            renderList(e.target.value);
            list.classList.remove("hidden");
        });

        document.addEventListener("click", (e) => {
            if (!wrapper.contains(e.target)) {
                list.classList.add("hidden");
                wrapper.classList.remove("active");
            }
        });

        return renderList; 
    }


    async function cargarEquipos() {
        try {
            const res = await fetch('/equipos');
            const datos = await res.json();
            
            if (datos.error) throw new Error(datos.error);
            const equipos = datos.equipos;

            
            const defaultLocal = equipos.find(e => e.toLowerCase().includes("strongest")) || equipos[0];
            const defaultVisitante = equipos.find(e => e.toLowerCase().includes("bolivar") || e.toLowerCase().includes("bolívar")) || equipos[1];

            setupCustomSelect(inputLocal, listLocal, wrapLocal, equipos, (team) => {
                ui.localName.textContent = team;
                fetchLogoWithDelay(team, imgLocal, spinnerLocal);
            });

            setupCustomSelect(inputVisitante, listVisitante, wrapVisitante, equipos, (team) => {
                ui.visitanteName.textContent = team;
                fetchLogoWithDelay(team, imgVisitante, spinnerVisitante);
            });

            if (defaultLocal) {
                inputLocal.value = defaultLocal;
                ui.localName.textContent = defaultLocal;
                fetchLogoWithDelay(defaultLocal, imgLocal, spinnerLocal);
            }

            if (defaultVisitante) {
                inputVisitante.value = defaultVisitante;
                ui.visitanteName.textContent = defaultVisitante;
                fetchLogoWithDelay(defaultVisitante, imgVisitante, spinnerVisitante);
            }

        } catch (e) {
            console.error("Error:", e);
            inputLocal.placeholder = "Error cargando datos";
        }
    }

    btn.addEventListener("click", async () => {
        const local = inputLocal.value;
        const visitante = inputVisitante.value;

        if (!local || !visitante || local === visitante) {
            alert("Selecciona dos equipos válidos y diferentes.");
            return;
        }

        ui.resultado.classList.add('hidden');
        ui.loading.classList.remove('hidden');
        btn.disabled = true;
        btn.style.opacity = "0.5";
        btn.innerHTML = "CALCULANDO...";

        ui.barLocal.style.width = "0%";
        ui.barEmpate.style.width = "0%";
        ui.barVisitante.style.width = "0%";

        try {
            const res = await fetch('/prediccion', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ local, visitante })
            });
            const datos = await res.json();
            
            if(datos.error) throw new Error(datos.error);

            ui.localVal.textContent = datos.prob_local + "%";
            ui.empateVal.textContent = datos.prob_empate + "%";
            ui.visitanteVal.textContent = datos.prob_visitante + "%";

            ui.loading.classList.add('hidden');
            ui.resultado.classList.remove('hidden');

            setTimeout(() => {
                ui.barLocal.style.width = datos.prob_local + "%";
                ui.barEmpate.style.width = datos.prob_empate + "%";
                ui.barVisitante.style.width = datos.prob_visitante + "%";
            }, 100);

        } catch (e) {
            alert("Error: " + e.message);
            ui.loading.classList.add('hidden');
        } finally {
            btn.disabled = false;
            btn.style.opacity = "1";
            btn.innerHTML = 'INICIAR SIMULACIÓN <i class="fas fa-chevron-right"></i>';
        }
    });

    cargarEquipos();
});