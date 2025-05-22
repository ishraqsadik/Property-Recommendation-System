// API endpoints
const API_URL = {
  recommendations: '/api/properties/recommend',
  feedback: '/api/feedback/submit',
  stats: '/api/feedback/stats',
  retrain: '/api/feedback/retrain',
  export: '/api/feedback/export',
  testSubjects: '/api/properties/test-subjects'
};

// Current state
let currentSubjectProperty = null;
let currentRecommendations = [];
let testProperties = [];

// Sample test property to use as fallback
const sampleTestProperties = [
  {
    id: "sample_1",
    address: "123 Sample St",
    structure_type: "Detached",
    bedrooms: 3,
    full_baths: 2,
    half_baths: 1,
    gla: 2000,
    lot_size: 5000,
    age: 10,
    city: "Sample City",
    province: "Ontario",
    postal_code: "12345",
    order_id: "sample_order_1"
  },
  {
    id: "sample_2",
    address: "456 Example Ave",
    structure_type: "Townhouse",
    bedrooms: 2,
    full_baths: 1,
    half_baths: 1,
    gla: 1500,
    lot_size: 2000,
    age: 5,
    city: "Example City",
    province: "Alberta",
    postal_code: "23456",
    order_id: "sample_order_2"
  }
];

// Initialize when document is ready
document.addEventListener('DOMContentLoaded', function() {
  console.log("Initializing application");
  initializeApp();
});

// Main initialization function
async function initializeApp() {
  // Setup form submission handler
  const propertyForm = document.getElementById('property-form');
  if (propertyForm) {
    propertyForm.addEventListener('submit', handleFormSubmit);
  }

  // Setup feedback buttons
  const approveButton = document.getElementById('approve-button');
  const rejectButton = document.getElementById('reject-button');
  
  if (approveButton) approveButton.addEventListener('click', () => submitFeedback(true));
  if (rejectButton) rejectButton.addEventListener('click', () => submitFeedback(false));
  
  // Load test subjects
  await loadTestSubjects();
}

// Load test subjects from the API
async function loadTestSubjects() {
  const dropdown = document.getElementById('test_property_select');
  if (!dropdown) return;
  
  try {
    showLoader(true);
    
    // Get test subjects from API
    const response = await fetch(API_URL.testSubjects);
    
    if (!response.ok) {
      throw new Error(`Failed to load test subjects: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    console.log(`Loaded ${data.length} test subjects`);
    
    // Store for later use
    testProperties = data;
    
    // Update model stats
    const statsElement = document.getElementById('model-stats');
    if (statsElement) {
      statsElement.textContent = `${data.length} test properties available`;
    }
    
    // Clear dropdown
    dropdown.innerHTML = '';
    
    // Add empty option
    const emptyOption = document.createElement('option');
    emptyOption.value = '';
    emptyOption.textContent = '-- Select a test property --';
    dropdown.appendChild(emptyOption);
    
    // Add options for each test subject
    data.forEach(property => {
      const option = document.createElement('option');
      option.value = property.id;
      
      // Create descriptive label
      const address = property.address || 'No address';
      const beds = property.bedrooms || '?';
      const baths = (property.full_baths || 0) + (property.half_baths ? 0.5 * property.half_baths : 0);
      const sqft = property.gla ? Math.round(property.gla) : '?';
      
      option.textContent = `${address} - ${beds}bd ${baths}ba ${sqft}sqft`;
      dropdown.appendChild(option);
    });
  } catch (error) {
    console.error("Failed to load test subjects:", error);
    showAlert(`Failed to load test subjects: ${error.message}`, 'danger');
    
    // Add error option
    dropdown.innerHTML = '<option value="">Error loading test properties</option>';
  } finally {
    showLoader(false);
  }
}

// Handle form submission
async function handleFormSubmit(event) {
  event.preventDefault();
  
  const dropdown = document.getElementById('test_property_select');
  if (!dropdown || !dropdown.value) {
    showAlert('Please select a test property first', 'warning');
    return;
  }
  
  const selectedId = dropdown.value;
  const selectedProperty = testProperties.find(p => p.id === selectedId);
  
  if (!selectedProperty) {
    showAlert('Selected property not found', 'danger');
    return;
  }
  
  try {
    showLoader(true);
    
    // Send API request to get recommendations
    const response = await fetch(API_URL.recommendations, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        subject_property: selectedProperty,
        max_recommendations: 3,
        min_score_threshold: 0.01
      })
    });
    
    if (!response.ok) {
      throw new Error(`Failed to get recommendations: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    
    // Store current data
    currentSubjectProperty = data.subject_property;
    currentRecommendations = data.recommendations || [];
    
    // Display results
    displaySubjectProperty(currentSubjectProperty);
    displayRecommendations(currentRecommendations);
    displayTrueCompSummary(currentRecommendations);
    
    // Show feedback section
    showFeedbackSection(true);
    
  } catch (error) {
    console.error("Error getting recommendations:", error);
    showAlert(error.message, 'danger');
  } finally {
    showLoader(false);
  }
}

// Display the subject property
function displaySubjectProperty(property) {
  const container = document.getElementById('subject-property-container');
  if (!container) return;
  
  container.innerHTML = '';
  
  if (property) {
    const card = createPropertyCard(property, 'subject-property');
    container.appendChild(card);
  }
}

// Display recommended properties
function displayRecommendations(properties) {
  const container = document.getElementById('recommendation-container');
  if (!container) return;
  
  container.innerHTML = '';
  
  if (!properties || properties.length === 0) {
    container.innerHTML = '<p>No recommendations found.</p>';
    return;
  }
  
  properties.forEach(property => {
    const card = createPropertyCard(property, 'recommendation-property');
    container.appendChild(card);
  });
}

// Display true comp summary
function displayTrueCompSummary(recommendations) {
  // Create or get summary container
  let summaryContainer = document.getElementById('true-comp-summary');
  if (!summaryContainer) {
    summaryContainer = document.createElement('div');
    summaryContainer.id = 'true-comp-summary';
    summaryContainer.className = 'true-comp-summary';
    
    // Insert after recommendation container
    const recommendationContainer = document.getElementById('recommendation-container');
    if (recommendationContainer && recommendationContainer.parentNode) {
      recommendationContainer.parentNode.insertBefore(summaryContainer, recommendationContainer.nextSibling);
    }
  }
  
  // Count true comps
  const trueComps = recommendations.filter(prop => prop.is_true_comp === 1);
  
  // Create summary content
  let summaryHTML = '';
  if (trueComps.length > 0) {
    summaryHTML = `
      <div class="summary-box true-comp-box">
        <h4>✓ True Comps Found</h4>
        <p>Found ${trueComps.length} true comparable property/properties in the recommendations:</p>
        <ul>
          ${trueComps.map(comp => `<li>${comp.address} (Score: ${comp.score.toFixed(3)})</li>`).join('')}
        </ul>
      </div>
    `;
  } else {
    summaryHTML = `
      <div class="summary-box no-comp-box">
        <h4>No True Comps Found</h4>
        <p>None of the recommended properties are true comparables from the original appraisal.</p>
      </div>
    `;
  }
  
  summaryContainer.innerHTML = summaryHTML;
}

// Create a property card element
function createPropertyCard(property, className) {
  const card = document.createElement('div');
  card.className = `property-card ${className || ''}`;
  card.dataset.id = property.id || '';
  
  // Check if this is a true comp
  const isTrueComp = property.is_true_comp === 1;
  
  // Get property details with fallbacks
  const price = property.sale_price 
    ? `$${Number(property.sale_price).toLocaleString()}`
    : 'Price not available';
    
  const address = property.address || 'Address not available';
  const beds = property.bedrooms || '?';
  const fullBaths = property.full_baths || 0;
  const halfBaths = property.half_baths || 0;
  const sqft = property.gla ? Math.round(property.gla) : '?';
  const structureType = property.structure_type || 'Not specified';
  const age = property.age || '?';
  const score = property.score ? property.score.toFixed(3) : '';
  
  // Create card HTML
  card.innerHTML = `
    <div class="property-details">
      <div class="property-header">
        <div class="property-price">${price}</div>
        ${score ? `<div class="property-score">Score: ${score}</div>` : ''}
        ${isTrueComp ? '<div class="true-comp-badge">✓ TRUE COMP</div>' : ''}
      </div>
      <div class="property-address">${address}</div>
      <div class="property-features">
        <span class="property-feature">${beds} Beds</span>
        <span class="property-feature">${fullBaths}.${halfBaths} Baths</span>
        <span class="property-feature">${sqft} sqft</span>
      </div>
      <div class="property-features">
        <span class="property-feature">Type: ${structureType}</span>
        <span class="property-feature">Age: ${age} yrs</span>
      </div>
      ${className === 'recommendation-property' ? `
        <div class="property-actions">
          <button class="btn btn-primary select-property-btn">Select</button>
        </div>
      ` : ''}
    </div>
  `;
  
  // Add click handler for selection
  const selectBtn = card.querySelector('.select-property-btn');
  if (selectBtn) {
    selectBtn.addEventListener('click', () => {
      card.classList.toggle('selected');
      selectBtn.textContent = card.classList.contains('selected') ? 'Unselect' : 'Select';
    });
  }
  
  return card;
}

// Submit feedback for recommendations
async function submitFeedback(isApproved) {
  if (!currentSubjectProperty || !currentRecommendations.length) {
    showAlert('No recommendations to provide feedback for', 'warning');
    return;
  }
  
  try {
    showLoader(true);
    
    // Get selected properties if not approved
    const selectedProperties = isApproved 
      ? [] 
      : Array.from(document.querySelectorAll('.property-card.selected'))
          .map(card => currentRecommendations.find(p => p.id === card.dataset.id))
          .filter(Boolean);
    
    // Get comments
    const comments = document.getElementById('feedback-comments')?.value || '';
    
    // Create feedback data
    const feedbackData = {
      subject_id: currentSubjectProperty.id,
      subject_data: currentSubjectProperty,
      recommended_properties: currentRecommendations,
      is_approved: isApproved,
      selected_properties: selectedProperties,
      comments: comments
    };
    
    // Send feedback to API
    const response = await fetch(API_URL.feedback, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify(feedbackData)
    });
    
    if (!response.ok) {
      throw new Error(`Failed to submit feedback: ${response.status} ${response.statusText}`);
    }
    
    const data = await response.json();
    
    // Show success message
    showAlert('Feedback submitted successfully!', 'success');
    
    // Reset UI
    resetUI();
    
  } catch (error) {
    console.error("Error submitting feedback:", error);
    showAlert(error.message, 'danger');
  } finally {
    showLoader(false);
  }
}

// Reset the UI
function resetUI() {
  // Clear form
  const form = document.getElementById('property-form');
  if (form) form.reset();
  
  // Clear properties
  document.getElementById('subject-property-container').innerHTML = '';
  document.getElementById('recommendation-container').innerHTML = '';
  
  // Hide feedback section
  showFeedbackSection(false);
  
  // Clear current data
  currentSubjectProperty = null;
  currentRecommendations = [];
}

// Show/hide the feedback section
function showFeedbackSection(show) {
  const section = document.getElementById('feedback-section');
  if (section) {
    section.style.display = show ? 'block' : 'none';
  }
}

// Show/hide loader
function showLoader(show) {
  const loader = document.getElementById('loader');
  if (loader) {
    loader.style.display = show ? 'block' : 'none';
  }
}

// Show alert message
function showAlert(message, type) {
  const alertContainer = document.getElementById('alert-container');
  if (!alertContainer) return;
  
  // Create alert element
  const alert = document.createElement('div');
  alert.className = `alert alert-${type}`;
  alert.textContent = message;
  
  // Add close button
  const closeButton = document.createElement('button');
  closeButton.innerHTML = '&times;';
  closeButton.style.float = 'right';
  closeButton.style.border = 'none';
  closeButton.style.background = 'transparent';
  closeButton.style.cursor = 'pointer';
  closeButton.onclick = () => alert.remove();
  
  alert.prepend(closeButton);
  
  // Add to container
  alertContainer.appendChild(alert);
  
  // Auto-remove after 5 seconds
  setTimeout(() => alert.remove(), 5000);
}

async function loadDashboardStats() {
  try {
    showLoader(true);
    
    // Fetch feedback statistics
    const response = await fetch(API_URL.stats);
    const stats = await response.json();
    
    if (!response.ok) {
      throw new Error(stats.detail || 'Failed to load statistics');
    }
    
    // Display statistics
    displayDashboardStats(stats);
    
  } catch (error) {
    showAlert(error.message, 'danger');
    console.error('Error:', error);
  } finally {
    showLoader(false);
  }
}

function displayDashboardStats(stats) {
  // Update stats cards
  document.getElementById('total-feedback-count').textContent = stats.total_count;
  document.getElementById('positive-feedback-count').textContent = stats.positive_count;
  document.getElementById('negative-feedback-count').textContent = stats.negative_count;
  document.getElementById('approval-rate').textContent = `${Math.round(stats.approval_rate * 100)}%`;
  
  // Update last updated timestamp
  if (stats.last_updated) {
    const lastUpdated = new Date(stats.last_updated);
    document.getElementById('last-updated-time').textContent = lastUpdated.toLocaleString();
  }
  
  // Update last retraining timestamp
  if (stats.last_retrain) {
    const lastRetrain = new Date(stats.last_retrain);
    document.getElementById('last-retrain-time').textContent = lastRetrain.toLocaleString();
  } else {
    document.getElementById('last-retrain-time').textContent = 'Never';
  }
}

async function handleRetrainModel() {
  try {
    showLoader(true);
    
    // Request model retraining
    const response = await fetch(API_URL.retrain, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        force: true
      }),
    });
    
    const result = await response.json();
    
    if (!response.ok) {
      throw new Error(result.detail || 'Failed to retrain model');
    }
    
    // Show success message
    showAlert(
      result.status === 'success' 
        ? 'Model retrained successfully!' 
        : result.reason || 'Retraining skipped',
      result.status === 'success' ? 'success' : 'info'
    );
    
    // Reload stats after retraining
    loadDashboardStats();
    
  } catch (error) {
    showAlert(error.message, 'danger');
    console.error('Error:', error);
  } finally {
    showLoader(false);
  }
}

async function handleExportFeedback() {
  try {
    showLoader(true);
    
    // Request feedback export
    const response = await fetch(API_URL.export);
    const result = await response.json();
    
    if (!response.ok) {
      throw new Error(result.detail || 'Failed to export feedback');
    }
    
    // Show success message
    showAlert(`Feedback data exported to ${result.path}`, 'success');
    
  } catch (error) {
    showAlert(error.message, 'danger');
    console.error('Error:', error);
  } finally {
    showLoader(false);
  }
}

// Debug function to directly test the API
async function debugTestPropertiesAPI() {
  console.log("Starting API debug...");
  
  try {
    // Create a modal to display debug info
    const modal = document.createElement('div');
    modal.style.position = 'fixed';
    modal.style.top = '50%';
    modal.style.left = '50%';
    modal.style.transform = 'translate(-50%, -50%)';
    modal.style.backgroundColor = 'white';
    modal.style.padding = '20px';
    modal.style.borderRadius = '5px';
    modal.style.boxShadow = '0 0 10px rgba(0,0,0,0.5)';
    modal.style.zIndex = '1000';
    modal.style.maxWidth = '80%';
    modal.style.maxHeight = '80%';
    modal.style.overflow = 'auto';
    
    modal.innerHTML = '<h3>API Debug Results</h3><pre id="debug-output">Loading...</pre><button id="close-debug">Close</button>';
    document.body.appendChild(modal);
    
    const outputArea = document.getElementById('debug-output');
    const closeButton = document.getElementById('close-debug');
    closeButton.addEventListener('click', () => modal.remove());
    
    // Try to fetch the API directly
    const response = await fetch(API_URL.testSubjects + '?limit=50');
    const responseText = await response.text();
    
    try {
      const data = JSON.parse(responseText);
      outputArea.textContent = JSON.stringify(data, null, 2);
    } catch (parseError) {
      outputArea.textContent = 'Error parsing JSON: ' + parseError + '\n\nRaw response:\n' + responseText;
    }
    
    // Add a section at the top for status info
    outputArea.textContent = `Status: ${response.status} ${response.statusText}\n\n` + outputArea.textContent;
    
    // Also log to console
    console.log("API Debug Results:", {
      status: response.status,
      statusText: response.statusText,
      responseText: responseText
    });
    
  } catch (error) {
    console.error("Debug error:", error);
    showAlert("API Debug error: " + error.message, "danger");
  }
}

// Helper function to populate the dropdown
function populateTestPropertyDropdown(selectElement, properties) {
  // Clear existing options
  selectElement.innerHTML = '';
  
  // Add default option
  const defaultOption = document.createElement('option');
  defaultOption.value = '';
  defaultOption.textContent = '-- Select a test property --';
  selectElement.appendChild(defaultOption);
  
  // Add each property to the dropdown
  properties.forEach(property => {
    const option = document.createElement('option');
    option.value = property.id;
    
    // Create descriptive label
    const address = property.address || 'No address';
    const price = property.sale_price ? `$${Number(property.sale_price).toLocaleString()}` : 'No price';
    const beds = property.bedrooms || '?';
    const baths = (property.full_baths || 0) + (property.half_baths ? 0.5 * property.half_baths : 0);
    const sqft = property.gla ? Math.round(property.gla) : '?';
    
    option.textContent = `${address} - ${price}, ${beds}bd ${baths}ba ${sqft}sqft`;
    selectElement.appendChild(option);
  });
}

// Export functions for use in other scripts
window.app = {
  handleRetrainModel,
  handleExportFeedback,
  resetUI,
  loadTestSubjects
}; 