// Global variables
let map;
let stationsLayer;
let historicalLayer;
let selectedStation = null;
let editMode = false;
let marker = null;

// Initialize the map
document.addEventListener('DOMContentLoaded', function() {
    console.log("DOM loaded - initializing map");
    
    // Create the map
    map = L.map('map').setView([52.516667, 13.388889], 12);
    
    // Add OpenStreetMap as the base layer
    L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors'
    }).addTo(map);
    
    // Set up event listeners
    document.getElementById('yearSelect').addEventListener('change', function() {
        loadFilterData();
    });
    
    document.getElementById('loadStationsBtn').addEventListener('click', loadStations);
    document.getElementById('mapLayerSelect').addEventListener('change', changeMapLayer);
    document.getElementById('editModeSwitch').addEventListener('change', toggleEditMode);
    document.getElementById('saveLocationBtn').addEventListener('click', saveStationLocation);
    document.getElementById('cancelEditBtn').addEventListener('click', cancelEdit);
    
    // Initial data loading
    loadFilterData();
});

// Load filter data (types and lines) for the selected year
function loadFilterData() {
    const year = document.getElementById('yearSelect').value;
    console.log(`Loading filter data for year ${year}`);
    
    // Add timestamp to prevent caching
    const apiUrl = `/api/filter-options/${year}?t=${Date.now()}`;
    
    fetch(apiUrl)
        .then(response => {
            console.log(`Filter API response status: ${response.status}`);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log(`Received filter data:`, data);
            updateFilterDropdowns(data);
        })
        .catch(error => {
            console.error('Error loading filter data:', error);
            alert('Failed to load filter options. Please check the console for details.');
        });
}

// Update filter dropdowns with received data
function updateFilterDropdowns(data) {
    // Update types dropdown
    const typeSelect = document.getElementById('stationTypeSelect');
    typeSelect.innerHTML = '<option value="all">All Types</option>';
    
    if (data && data.types && Array.isArray(data.types)) {
        data.types.forEach(type => {
            const option = document.createElement('option');
            option.value = type;
            option.textContent = type;
            typeSelect.appendChild(option);
        });
        console.log(`Added ${data.types.length} type options`);
    }
    
    // Update lines dropdown
    const lineSelect = document.getElementById('lineSelect');
    lineSelect.innerHTML = '<option value="all">All Lines</option>';
    
    if (data && data.lines && Array.isArray(data.lines)) {
        data.lines.forEach(line => {
            const option = document.createElement('option');
            option.value = line;
            option.textContent = line;
            lineSelect.appendChild(option);
        });
        console.log(`Added ${data.lines.length} line options`);
    }
}

// Load stations based on selected filters
function loadStations() {
    const year = document.getElementById('yearSelect').value;
    const stationType = document.getElementById('stationTypeSelect').value;
    const line = document.getElementById('lineSelect').value;
    const eastWest = document.getElementById('eastWestSelect').value;
    
    console.log(`Loading stations for year=${year}, type=${stationType}, line=${line}, east_west=${eastWest}`);
    
    // Clear existing stations layer
    if (stationsLayer) {
        map.removeLayer(stationsLayer);
    }
    
    // Build URL with filters
    let apiUrl = `/api/stations/${year}`;
    const params = [];
    
    if (stationType !== 'all') {
        params.push(`type=${encodeURIComponent(stationType)}`);
    }
    
    if (line !== 'all') {
        params.push(`line=${encodeURIComponent(line)}`);
    }
    
    if (eastWest !== 'all') {
        params.push(`east_west=${encodeURIComponent(eastWest)}`);
    }
    
    if (params.length > 0) {
        apiUrl += '?' + params.join('&');
    }
    
    console.log(`Fetching stations from: ${apiUrl}`);
    
    // Fetch stations
    fetch(apiUrl)
        .then(response => {
            console.log(`Stations API response status: ${response.status}`);
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(data => {
            console.log(`Received stations data with ${data.features?.length || 0} features`);
            displayStations(data);
        })
        .catch(error => {
            console.error('Error loading stations:', error);
            alert('Failed to load stations. Please check the console for details.');
        });
}

// Display stations on the map
function displayStations(geojson) {
    // Create a GeoJSON layer
    stationsLayer = L.geoJSON(geojson, {
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
            } else if (stationType === 'tram' || stationType === 'strassenbahn') {
                markerColor = '#ffa500'; // Orange for Tram
            } else if (stationType === 'ferry') {
                markerColor = '#00FFFF'; // Cyan for Ferry
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
    if (geojson.features && geojson.features.length > 0) {
        map.fitBounds(stationsLayer.getBounds());
    }
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
    
    console.log(`Loading map layer: ${selectedLayer}`);
    
    // Get metadata for the selected map
    fetch(`/api/map_metadata/${selectedLayer}`)
        .then(response => {
            if (!response.ok) {
                throw new Error(`HTTP error! Status: ${response.status}`);
            }
            return response.json();
        })
        .then(metadata => {
            console.log(`Map metadata:`, metadata);
            
            // Load the map as a GeoTIFF layer
            fetch(`/api/map/${selectedLayer}`)
                .then(response => {
                    if (!response.ok) {
                        throw new Error(`HTTP error! Status: ${response.status}`);
                    }
                    return response.arrayBuffer();
                })
                .then(arrayBuffer => {
                    console.log(`Received map file of size: ${arrayBuffer.byteLength} bytes`);
                    
                    // Parse the georaster
                    return parseGeoraster(arrayBuffer);
                })
                .then(georaster => {
                    console.log(`Parsed georaster:`, georaster);
                    
                    // Create the layer
                    historicalLayer = new GeoRasterLayer({
                        georaster: georaster,
                        opacity: 0.7,
                        resolution: 256
                    });
                    
                    // Add to map
                    historicalLayer.addTo(map);
                    
                    // Fit to bounds if available
                    if (metadata.bounds) {
                        const bounds = [
                            [metadata.bounds[1], metadata.bounds[0]], // Southwest corner
                            [metadata.bounds[3], metadata.bounds[2]]  // Northeast corner
                        ];
                        map.fitBounds(bounds);
                    }
                })
                .catch(error => {
                    console.error('Error loading map layer:', error);
                    alert('Failed to load map layer: ' + error.message);
                });
        })
        .catch(error => {
            console.error('Error getting map metadata:', error);
            alert('Failed to get map metadata: ' + error.message);
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
    
    console.log(`Edit mode ${editMode ? 'enabled' : 'disabled'}`);
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
    
    console.log(`Saving new location for station ${stationId}: ${newLatLng.lat}, ${newLatLng.lng}`);
    
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
    .then(response => {
        if (!response.ok) {
            throw new Error(`HTTP error! Status: ${response.status}`);
        }
        return response.json();
    })
    .then(data => {
        console.log(`Update response:`, data);
        
        if (data.status === 'success') {
            alert('Station location updated successfully!');
            
            // Instead of reloading all stations, just update the marker and GeoJSON
            if (stationsLayer) {
                stationsLayer.eachLayer(function(layer) {
                    // Find the layer for this station
                    const props = layer.feature.properties;
                    const layerStationId = props.stop_id || props.id;
                    
                    if (layerStationId == stationId) {
                        // Update the coordinates in the layer's feature
                        layer.feature.geometry.coordinates = [newLatLng.lng, newLatLng.lat];
                        
                        // Update the layer's position
                        layer.setLatLng(newLatLng);
                    }
                });
            }
            
            // Clear the draggable marker
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
        console.error('Error updating station:', error);
        alert('An error occurred while updating the station location: ' + error.message);
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