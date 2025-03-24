// Load filter options (types and lines) for the selected year

function loadFilterOptions() {
    const year = document.getElementById('yearSelect').value;
    
    console.log("DEBUG: Loading filter options for year:", year);
    const apiUrl = `/api/filter-options/${year}`;
    console.log("DEBUG: API URL:", apiUrl);
    
    // Add timestamp to prevent caching
    const urlWithCache = `${apiUrl}?t=${new Date().getTime()}`;
    
    fetch(urlWithCache)
        .then(response => {
            console.log("DEBUG: Filter options response status:", response.status);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log("DEBUG: Received filter options data:", data);
            
            if (!data || typeof data !== 'object') {
                console.error("Invalid response format - expected object");
                return;
            }
            
            // Check if types array exists
            if (Array.isArray(data.types)) {
                // Update type select
                const typeSelect = document.getElementById('stationTypeSelect');
                // Clear existing options except the first one
                while (typeSelect.options.length > 1) {
                    typeSelect.remove(1);
                }
                
                // Add new type options
                data.types.forEach(type => {
                    const option = document.createElement('option');
                    option.value = type;
                    option.textContent = type;
                    typeSelect.appendChild(option);
                });
                
                console.log("DEBUG: Added", data.types.length, "type options");
            } else {
                console.error("Types array missing from response");
            }
            
            // Check if lines array exists
            if (Array.isArray(data.lines)) {
                // Update line select
                const lineSelect = document.getElementById('lineSelect');
                // Clear existing options except the first one
                while (lineSelect.options.length > 1) {
                    lineSelect.remove(1);
                }
                
                // Add new line options
                data.lines.forEach(line => {
                    const option = document.createElement('option');
                    option.value = line;
                    option.textContent = line;
                    lineSelect.appendChild(option);
                });
                
                console.log("DEBUG: Added", data.lines.length, "line options");
            } else {
                console.error("Lines array missing from response");
            }
        })
        .catch(error => {
            console.error('Error loading filter options:', error);
            
            // Fallback to direct debug call
            console.log("Attempting fallback debug API call...");
            fetch(`/debug-api/types/${year}`)
                .then(response => response.json())
                .then(debugData => {
                    console.log("Debug API response:", debugData);
                })
                .catch(debugError => {
                    console.error("Debug API error:", debugError);
                });
        });
}

// Debug function to examine data structure
function debugDataStructure() {
    const year = document.getElementById('yearSelect').value;
    
    // Create a modal to show debug information
    const modal = document.createElement('div');
    modal.className = 'modal fade';
    modal.id = 'debugModal';
    modal.tabIndex = '-1';
    modal.setAttribute('aria-labelledby', 'debugModalLabel');
    modal.setAttribute('aria-hidden', 'true');
    
    modal.innerHTML = `
        <div class="modal-dialog modal-lg">
            <div class="modal-content">
                <div class="modal-header">
                    <h5 class="modal-title" id="debugModalLabel">Data Structure Debug</h5>
                    <button type="button" class="btn-close" data-bs-dismiss="modal" aria-label="Close"></button>
                </div>
                <div class="modal-body">
                    <h6>Loading data structure for year ${year}...</h6>
                    <div id="debugContent" style="max-height: 400px; overflow-y: auto;"></div>
                </div>
                <div class="modal-footer">
                    <button type="button" class="btn btn-secondary" data-bs-dismiss="modal">Close</button>
                </div>
            </div>
        </div>
    `;
    
    document.body.appendChild(modal);
    
    // Show the modal
    const modalInstance = new bootstrap.Modal(document.getElementById('debugModal'));
    modalInstance.show();
    
    // Get debug information
    const debugContent = document.getElementById('debugContent');
    
    // 1. Get the types
    fetch(`/debug-api/types/${year}`)
        .then(response => response.json())
        .then(data => {
            debugContent.innerHTML += `
                <div class="card mb-3">
                    <div class="card-header">Types and Counts</div>
                    <div class="card-body">
                        <p>Total Stations: ${data.total_stations}</p>
                        <h6>Type Counts:</h6>
                        <ul>
                            ${Object.entries(data.type_counts).map(([type, count]) => 
                                `<li>${type}: ${count}</li>`).join('')}
                        </ul>
                    </div>
                </div>
            `;
            
            // 2. Check data folders
            return fetch(`/debug-api/folder/processed`);
        })
        .then(response => response.json())
        .then(data => {
            // Filter folders for the current year
            const yearFolders = data.folders.filter(folder => folder.name.startsWith(year));
            
            debugContent.innerHTML += `
                <div class="card mb-3">
                    <div class="card-header">Folders for Year ${year}</div>
                    <div class="card-body">
                        <ul>
                            ${yearFolders.map(folder => 
                                `<li>${folder.name} <button class="btn btn-sm btn-outline-primary check-folder" data-path="${folder.path}">Check Contents</button></li>`).join('')}
                        </ul>
                    </div>
                </div>
            `;
            
            // Add event listeners to the check folder buttons
            document.querySelectorAll('.check-folder').forEach(button => {
                button.addEventListener('click', function() {
                    const path = this.getAttribute('data-path');
                    
                    fetch(`/debug-api/folder/${path}`)
                        .then(response => response.json())
                        .then(folderData => {
                            // Find the stops.csv file
                            const stopsFile = folderData.files.find(file => file.name === 'stops.csv');
                            
                            if (stopsFile) {
                                return fetch(`/debug-api/file/${stopsFile.path}`);
                            } else {
                                throw new Error('stops.csv not found in folder');
                            }
                        })
                        .then(response => response.json())
                        .then(fileData => {
                            // Create a new card for the file data
                            const fileCard = document.createElement('div');
                            fileCard.className = 'card mb-3';
                            fileCard.innerHTML = `
                                <div class="card-header">File: ${fileData.filename}</div>
                                <div class="card-body">
                                    <h6>Columns (${fileData.columns.length}):</h6>
                                    <p>${fileData.columns.join(', ')}</p>
                                    <h6>Sample Rows (${fileData.total_rows} total):</h6>
                                    <div style="overflow-x: auto;">
                                        <table class="table table-sm">
                                            <thead>
                                                <tr>
                                                    ${fileData.columns.map(col => `<th>${col}</th>`).join('')}
                                                </tr>
                                            </thead>
                                            <tbody>
                                                ${fileData.rows.map(row => `
                                                    <tr>
                                                        ${fileData.columns.map(col => `<td>${row[col] !== null ? row[col] : ''}</td>`).join('')}
                                                    </tr>
                                                `).join('')}
                                            </tbody>
                                        </table>
                                    </div>
                                </div>
                            `;
                            
                            debugContent.appendChild(fileCard);
                        })
                        .catch(error => {
                            alert('Error checking folder: ' + error.message);
                        });
                });
            });
        })
        .catch(error => {
            debugContent.innerHTML += `
                <div class="alert alert-danger">
                    Error: ${error.message}
                </div>
            `;
        });
}// Global variables
let map;
let stationsLayer;
let historicalLayer;
let selectedStation = null;
let editMode = false;
let marker = null;

// Initialize the map
document.addEventListener('DOMContentLoaded', function() {
    // Create the map
    map = L.map('map').setView([52.516667, 13.388889], 12);
    
    // Add OpenStreetMap as the base layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);
    
    // Set up event listeners
    document.getElementById('loadStationsBtn').addEventListener('click', loadStations);
    document.getElementById('mapLayerSelect').addEventListener('change', changeMapLayer);
    document.getElementById('exportGeoJSONBtn').addEventListener('click', exportGeoJSON);
    document.getElementById('editModeSwitch').addEventListener('change', toggleEditMode);
    document.getElementById('saveLocationBtn').addEventListener('click', saveStationLocation);
    document.getElementById('cancelEditBtn').addEventListener('click', cancelEdit);
    
    // If we have a default year, load stations for it
    if (document.getElementById('yearSelect').options.length > 0) {
        loadStations();
    }
});

// Load types for the selected year
function loadTypes() {
    const year = document.getElementById('yearSelect').value;
    
    console.log("Loading types for year:", year);
    fetch(`/api/types/${year}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(types => {
            console.log("Received types:", types);
            const typeSelect = document.getElementById('stationTypeSelect');
            
            // Clear existing options except the first one
            while (typeSelect.options.length > 1) {
                typeSelect.remove(1);
            }
            
            // Add new options
            for (const type of types) {
                const option = document.createElement('option');
                option.value = type;
                option.textContent = type;
                typeSelect.appendChild(option);
            }
        })
        .catch(error => {
            console.error('Error loading types:', error);
            // Try the debug endpoint to see what's happening
            fetch(`/debug-api/types/${year}`)
                .then(response => response.json())
                .then(data => {
                    console.log("Debug types data:", data);
                })
                .catch(debugError => {
                    console.error("Debug endpoint error:", debugError);
                });
        });
}

// Load lines for the selected year
function loadLines() {
    const year = document.getElementById('yearSelect').value;
    
    console.log("Loading lines for year:", year);
    fetch(`/api/lines/${year}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(lines => {
            console.log("Received lines:", lines);
            const lineSelect = document.getElementById('lineSelect');
            
            // Clear existing options except the first one
            while (lineSelect.options.length > 1) {
                lineSelect.remove(1);
            }
            
            // Add new options
            for (const line of lines) {
                const option = document.createElement('option');
                option.value = line;
                option.textContent = line;
                lineSelect.appendChild(option);
            }
        })
        .catch(error => {
            console.error('Error loading lines:', error);
        });
}

// Load stations for the selected year
function loadStations() {
    const year = document.getElementById('yearSelect').value;
    const stationType = document.getElementById('stationTypeSelect').value;
    const eastWest = document.getElementById('eastWestSelect').value;
    const line = document.getElementById('lineSelect').value;
    
    // Clear existing markers
    if (stationsLayer) {
        map.removeLayer(stationsLayer);
    }
    
    // Build the URL with filters
    let url = `/api/stations/${year}`;
    const params = [];
    
    if (stationType !== 'all') {
        params.push(`type=${stationType}`);
    }
    
    if (eastWest !== 'all') {
        params.push(`east_west=${eastWest}`);
    }
    
    if (line !== 'all') {
        params.push(`line=${encodeURIComponent(line)}`);
    }
    
    if (params.length > 0) {
        url += '?' + params.join('&');
    }
    
    console.log("Loading stations from URL:", url);
    
    // Fetch the stations
    fetch(url)
        .then(response => response.json())
        .then(data => {
            // Create a GeoJSON layer
            stationsLayer = L.geoJSON(data, {
                pointToLayer: function(feature, latlng) {
                    // Create a marker with style based on station type
                    const stationType = feature.properties.type;
                    let markerColor = '#3388ff'; // Default blue
                    
                    if (stationType === 'u-bahn') {
                        markerColor = '#0000ff'; // Blue for U-Bahn
                    } else if (stationType === 's-bahn') {
                        markerColor = '#008000'; // Green for S-Bahn
                    } else if (stationType === 'bus' || stationType === 'autobus') {
                        markerColor = '#ff0000'; // Red for Bus
                    } else if (stationType === 'strassenbahn' || stationType === 'tram') {
                        markerColor = '#ffa500'; // Orange for Tram
                    }
                    
                    // Check east/west and make the marker darker if it's east
                    const eastWest = feature.properties.east_west;
                    if (eastWest === 'east') {
                        markerColor = darkenColor(markerColor, 0.3);
                    }
                    
                    return L.circleMarker(latlng, {
                        radius: 6,
                        fillColor: markerColor,
                        color: '#000',
                        weight: 1,
                        opacity: 1,
                        fillOpacity: 0.8
                    });
                },
                onEachFeature: function(feature, layer) {
                    // Add popup with station name
                    const stationName = feature.properties.name || 
                                        feature.properties.stop_name || 
                                        feature.properties.node_label ||
                                        'Unnamed Station';
                    
                    layer.bindTooltip(stationName);
                    
                    // Add click event
                    layer.on('click', function(e) {
                        selectStation(feature, layer, e.latlng);
                    });
                }
            }).addTo(map);
            
            // Fit the map to the stations bounds if there are stations
            if (data.features && data.features.length > 0) {
                map.fitBounds(stationsLayer.getBounds());
            }
        })
        .catch(error => {
            console.error('Error loading stations:', error);
            alert('Failed to load stations. See console for details.');
        });
}

// Helper function to darken a color
function darkenColor(hex, factor) {
    // Convert hex to RGB
    let r = parseInt(hex.substring(1, 3), 16);
    let g = parseInt(hex.substring(3, 5), 16);
    let b = parseInt(hex.substring(5, 7), 16);
    
    // Darken
    r = Math.floor(r * (1 - factor));
    g = Math.floor(g * (1 - factor));
    b = Math.floor(b * (1 - factor));
    
    // Convert back to hex
    return '#' + ((1 << 24) + (r << 16) + (g << 8) + b).toString(16).slice(1);
}

// Change the historical map layer
function changeMapLayer() {
    const selectedLayer = document.getElementById('mapLayerSelect').value;
    
    // Remove current historical layer if it exists
    if (historicalLayer) {
        map.removeLayer(historicalLayer);
        historicalLayer = null;
    }
    
    // If "none" is selected, do nothing further
    if (selectedLayer === 'none') {
        return;
    }
    
    // Get metadata for the selected map
    fetch(`/api/map_metadata/${selectedLayer}`)
        .then(response => response.json())
        .then(metadata => {
            // Load the map as a GeoTIFF layer
                            fetch(`/api/map/${selectedLayer}`)
                .then(response => response.arrayBuffer())
                .then(arrayBuffer => {
                    // Check if parseGeoraster and GeoRasterLayer are available
                    if (!window.parseGeoraster) {
                        console.error("parseGeoraster is not defined. Make sure the georaster library is loaded correctly.");
                        alert("Error: Geospatial libraries not loaded correctly. Try refreshing the page.");
                        return;
                    }
                    
                    if (!window.GeoRasterLayer && !L.GeoRasterLayer) {
                        console.error("GeoRasterLayer is not defined. Make sure the georaster-layer-for-leaflet library is loaded correctly.");
                        alert("Error: Geospatial libraries not loaded correctly. Try refreshing the page.");
                        return;
                    }
                    
                    // Use the global function or fallback
                    const parseFunction = window.parseGeoraster || window.GeoRaster.parseGeoRaster;
                    const GeoRasterLayerClass = window.GeoRasterLayer || L.GeoRasterLayer;
                    
                    parseFunction(arrayBuffer).then(georaster => {
                        historicalLayer = new GeoRasterLayerClass({
                            georaster: georaster,
                            opacity: 0.7,
                            resolution: 256
                        });
                        
                        historicalLayer.addTo(map);
                        
                        // If we have bounds, fit to them
                        if (metadata.bounds) {
                            const bounds = [
                                [metadata.bounds[1], metadata.bounds[0]], // Southwest corner
                                [metadata.bounds[3], metadata.bounds[2]]  // Northeast corner
                            ];
                            map.fitBounds(bounds);
                        }
                    });
                })
                .catch(error => {
                    console.error('Error loading map layer:', error);
                    alert('Failed to load map layer. The file may be too large or in an unsupported format.');
                });
        })
        .catch(error => {
            console.error('Error getting map metadata:', error);
            alert('Failed to get map metadata.');
        });
}

// Select a station to view/edit
function selectStation(feature, layer, latlng) {
    selectedStation = feature;
    
    // Display station information
    const stationInfo = document.getElementById('stationInfo');
    const stationDetails = document.getElementById('stationDetails');
    
    // Extract station properties
    const properties = feature.properties;
    const stationName = properties.name || properties.stop_name || properties.node_label || 'Unnamed Station';
    const stationType = properties.type || 'Unknown';
    const stationId = properties.stop_id || properties.id || '';
    const eastWest = properties.east_west || 'Unknown';
    
    // Build HTML for station details
    let detailsHTML = `
        <p><strong>Name:</strong> ${stationName}</p>
        <p><strong>ID:</strong> ${stationId}</p>
        <p><strong>Type:</strong> ${stationType}</p>
        <p><strong>East/West:</strong> ${eastWest}</p>
        <p><strong>Coordinates:</strong> ${latlng.lat.toFixed(6)}, ${latlng.lng.toFixed(6)}</p>
    `;
    
    // Add any other relevant properties
    for (const key in properties) {
        if (!['name', 'stop_name', 'node_label', 'type', 'stop_id', 'id', 'east_west'].includes(key)) {
            detailsHTML += `<p><strong>${key}:</strong> ${properties[key]}</p>`;
        }
    }
    
    stationDetails.innerHTML = detailsHTML;
    stationInfo.style.display = 'block';
    
    // If edit mode is on, allow dragging
    if (editMode) {
        // If we already have a marker, remove it
        if (marker) {
            map.removeLayer(marker);
        }
        
        // Create a draggable marker
        marker = L.marker(latlng, { draggable: true }).addTo(map);
        
        // Update coordinates display when marker is dragged
        marker.on('drag', function(e) {
            const newLatlng = e.target.getLatLng();
            document.getElementById('stationDetails').innerHTML = detailsHTML.replace(
                `<p><strong>Coordinates:</strong> ${latlng.lat.toFixed(6)}, ${latlng.lng.toFixed(6)}</p>`,
                `<p><strong>Coordinates:</strong> ${newLatlng.lat.toFixed(6)}, ${newLatlng.lng.toFixed(6)}</p>`
            );
        });
    }
}

// Toggle edit mode
function toggleEditMode() {
    editMode = document.getElementById('editModeSwitch').checked;
    
    // Clear any existing selection when toggling mode
    if (marker) {
        map.removeLayer(marker);
        marker = null;
    }
    
    if (selectedStation) {
        document.getElementById('stationInfo').style.display = 'none';
        selectedStation = null;
    }
    
    // Update cursor style
    if (editMode && stationsLayer) {
        stationsLayer.eachLayer(function(layer) {
            layer.getElement().style.cursor = 'move';
        });
    } else if (stationsLayer) {
        stationsLayer.eachLayer(function(layer) {
            layer.getElement().style.cursor = '';
        });
    }
}

// Save the updated station location
function saveStationLocation() {
    if (!selectedStation || !marker) {
        alert('No station selected or no location change.');
        return;
    }
    
    const newLatLng = marker.getLatLng();
    const stationId = selectedStation.properties.stop_id || selectedStation.properties.id;
    const year = document.getElementById('yearSelect').value;
    
    if (!stationId) {
        alert('Cannot identify station ID.');
        return;
    }
    
    // Send the update to the server
    fetch('/api/update_station', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            id: stationId,
            lat: newLatLng.lat,
            lng: newLatLng.lng,
            year: year
        }),
    })
    .then(response => response.json())
    .then(data => {
        if (data.status === 'success') {
            alert('Station location updated successfully!');
            
            // Reload the stations to show the updated location
            loadStations();
            
            // Clear the selection
            if (marker) {
                map.removeLayer(marker);
                marker = null;
            }
            document.getElementById('stationInfo').style.display = 'none';
            selectedStation = null;
        } else {
            alert(`Failed to update station: ${data.message}`);
        }
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while updating the station location.');
    });
}

// Cancel editing and clear selection
function cancelEdit() {
    if (marker) {
        map.removeLayer(marker);
        marker = null;
    }
    
    document.getElementById('stationInfo').style.display = 'none';
    selectedStation = null;
}

// Export the current stations as GeoJSON
function exportGeoJSON() {
    const year = document.getElementById('yearSelect').value;
    
    // Open the export API in a new tab
    window.open(`/api/export_geojson/${year}`, '_blank');
    
    // Alternatively, you could use a fetch request and handle the download in JavaScript
    // but the simple approach above works for most cases
}