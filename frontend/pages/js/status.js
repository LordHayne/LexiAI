/**
 * Status module for Lexi AI
 */

// Update a status indicator element
export function updateStatusIndicator(indicatorId, textId, isOk, statusText) {
    const indicator = document.getElementById(indicatorId);
    const text = document.getElementById(textId);
    
    if (isOk) {
        indicator.classList.add('status-ok');
        indicator.classList.remove('status-error', 'status-warning');
    } else {
        indicator.classList.add('status-error');
        indicator.classList.remove('status-ok', 'status-warning');
    }
    
    text.innerText = statusText;
}

// Check system status by calling the health endpoint
export function checkSystemStatus() {
    console.log('Checking system status...');
    
    // Set initial status to checking
    updateStatusIndicator('llm-status', 'llm-status-text', null, 'Wird geprüft...');
    updateStatusIndicator('embedding-status', 'embedding-status-text', null, 'Wird geprüft...');
    updateStatusIndicator('database-status', 'database-status-text', null, 'Wird geprüft...');
    updateStatusIndicator('dimension-status', 'dimension-status-text', null, 'Wird geprüft...');
    
    // Fetch health status from API
    fetch('/v1/health')
        .then(response => response.json())
        .then(data => {
            console.log('Health data:', data);
            
            // Update status indicators based on health check
            if (data.status) {
                // LLM Status
                updateStatusIndicator(
                    'llm-status', 
                    'llm-status-text', 
                    data.components?.llm_service?.status === 'ok', 
                    data.components?.llm_service?.status === 'ok' ? 'Verbunden' : 'Nicht verbunden'
                );
                
                // Embedding Status
                updateStatusIndicator(
                    'embedding-status', 
                    'embedding-status-text', 
                    data.components?.embedding_service?.status === 'ok', 
                    data.components?.embedding_service?.status === 'ok' ? 'Verbunden' : 'Nicht verbunden'
                );
                
                // Database Status
                updateStatusIndicator(
                    'database-status', 
                    'database-status-text', 
                    data.components?.database?.status === 'ok', 
                    data.components?.database?.status === 'ok' ? 'Verbunden' : 'Nicht verbunden'
                );
                
                // Dimensions Status
                const dimensionsComponent = data.components?.dimensions || {};
                const dimensionStatus = dimensionsComponent.status === 'ok';
                const dimensionMessage = dimensionStatus 
                    ? `Kompatibel (${dimensionsComponent.actual || 0})` 
                    : (dimensionsComponent.message || 'Nicht kompatibel');
                
                updateStatusIndicator(
                    'dimension-status', 
                    'dimension-status-text', 
                    dimensionStatus, 
                    dimensionMessage
                );
            } else {
                // If health check failed, show all as error
                updateStatusIndicator('llm-status', 'llm-status-text', false, 'Status unbekannt');
                updateStatusIndicator('embedding-status', 'embedding-status-text', false, 'Status unbekannt');
                updateStatusIndicator('database-status', 'database-status-text', false, 'Status unbekannt');
                updateStatusIndicator('dimension-status', 'dimension-status-text', false, 'Status unbekannt');
            }
        })
        .catch(error => {
            console.error('Error checking health:', error);
            // If error, show all as error
            updateStatusIndicator('llm-status', 'llm-status-text', false, 'Verbindungsfehler');
            updateStatusIndicator('embedding-status', 'embedding-status-text', false, 'Verbindungsfehler');
            updateStatusIndicator('database-status', 'database-status-text', false, 'Verbindungsfehler');
            updateStatusIndicator('dimension-status', 'dimension-status-text', false, 'Verbindungsfehler');
        });
}

export function initializeStatus() {
    // Check system status
    checkSystemStatus();
}
