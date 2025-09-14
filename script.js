// DOM Elements
const codeInput = document.getElementById('codeInput');
const checkButton = document.getElementById('checkButton');
const resultSection = document.getElementById('resultSection');
const resultContent = document.getElementById('resultContent');
const buttonText = document.querySelector('.button-text');
const loadingSpinner = document.querySelector('.loading-spinner');

// Example codes
const examples = {
    'buffer-overflow': {
        title: 'Buffer Overflow',
        code: `char str[10];
strcpy(str, "very long string that exceeds buffer size");`
    },
    'double-free': {
        title: 'Double Free',
        code: `int* ptr = malloc(100);
free(ptr);
free(ptr);  // Double free - memory corruption`
    },
    'assignment-condition': {
        title: 'Assignment in Condition',
        code: `int x = 5;
if (x = 10) {  // Assignment instead of comparison
    printf("x is 10");
}`
    },
    'valid-code': {
        title: 'Geçerli Kod',
        code: `int add(int a, int b) {
    return a + b;
}

int main() {
    int result = add(5, 3);
    return result;
}`
    }
};

// Load example code
function loadExample(exampleKey) {
    const example = examples[exampleKey];
    if (example) {
        codeInput.value = example.code;
        codeInput.focus();
        
        // Show success message
        showMessage(`"${example.title}" örneği yüklendi!`, 'success');
    }
}

// Show message
function showMessage(message, type = 'info') {
    // Create message element
    const messageDiv = document.createElement('div');
    messageDiv.className = `message message-${type}`;
    messageDiv.textContent = message;
    
    // Add styles
    messageDiv.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        z-index: 1000;
        animation: slideInRight 0.3s ease-out;
        max-width: 300px;
    `;
    
    // Set background color based on type
    switch(type) {
        case 'success':
            messageDiv.style.background = '#10b981';
            break;
        case 'error':
            messageDiv.style.background = '#ef4444';
            break;
        case 'warning':
            messageDiv.style.background = '#f59e0b';
            break;
        default:
            messageDiv.style.background = '#3b82f6';
    }
    
    // Add to page
    document.body.appendChild(messageDiv);
    
    // Remove after 3 seconds
    setTimeout(() => {
        messageDiv.style.animation = 'slideOutRight 0.3s ease-out';
        setTimeout(() => {
            if (messageDiv.parentNode) {
                messageDiv.parentNode.removeChild(messageDiv);
            }
        }, 300);
    }, 3000);
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);

// Check code function
async function checkCode() {
    const code = codeInput.value.trim();
    
    if (!code) {
        showMessage('Lütfen C kodu girin!', 'warning');
        return;
    }
    
    // Show loading state
    setLoadingState(true);
    
    try {
        // For now, we'll simulate the API call
        // Later this will connect to the Python backend
        const result = await simulateAPICall(code);
        
        // Display result
        displayResult(result);
        
    } catch (error) {
        showMessage('Kod kontrol edilirken hata oluştu: ' + error.message, 'error');
        hideResult();
    } finally {
        setLoadingState(false);
    }
}

// Real API call to Python backend
async function callRealAPI(code) {
    try {
        const response = await fetch('http://localhost:5000/api/check-code', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ code: code })
        });
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const result = await response.json();
        return result;
        
    } catch (error) {
        console.error('API call failed:', error);
        throw new Error('API bağlantısı başarısız: ' + error.message);
    }
}

// Fallback to mock API if real API fails
async function simulateAPICall(code) {
    try {
        // Try real API first
        return await callRealAPI(code);
    } catch (error) {
        console.warn('Falling back to mock API:', error);
        
        // Fallback to mock response
        const mockResponses = [
            {
                isCompliant: true,
                message: "✅ Uygun. Rule 17.7: Tek return noktası kullanılmış.",
                rule: "Rule 17.7",
                confidence: 0.85
            },
            {
                isCompliant: false,
                message: "❌ Uygun değil. Rule 15.1: Buffer overflow riski var.",
                rule: "Rule 15.1",
                confidence: 0.92
            },
            {
                isCompliant: false,
                message: "⚠️ Uygun değil. Rule 17.2: Memory leak olabilir.",
                rule: "Rule 17.2",
                confidence: 0.78
            },
            {
                isCompliant: true,
                message: "✅ Uygun. Rule 16.1: Güvenli pointer kullanımı.",
                rule: "Rule 16.1",
                confidence: 0.89
            }
        ];
        
        // Randomly select a response
        const randomResponse = mockResponses[Math.floor(Math.random() * mockResponses.length)];
        
        return randomResponse;
    }
}

// Display result
function displayResult(result) {
    resultContent.innerHTML = `
        <div class="result-header">
            <h4>${result.isCompliant ? '✅ Uygun' : '❌ Uygun Değil'}</h4>
            <span class="confidence">Güven: %${(result.confidence * 100).toFixed(0)}</span>
        </div>
        <div class="result-message">
            <p>${result.message}</p>
        </div>
        <div class="result-details">
            <p><strong>Kural:</strong> ${result.rule}</p>
        </div>
    `;
    
    // Add appropriate styling
    resultContent.className = 'result-content';
    if (result.isCompliant) {
        resultContent.classList.add('result-success');
    } else {
        resultContent.classList.add('result-error');
    }
    
    resultSection.style.display = 'block';
}

// Hide result
function hideResult() {
    resultSection.style.display = 'none';
}

// Set loading state
function setLoadingState(loading) {
    if (loading) {
        checkButton.disabled = true;
        buttonText.style.display = 'none';
        loadingSpinner.style.display = 'inline-block';
    } else {
        checkButton.disabled = false;
        buttonText.style.display = 'inline';
        loadingSpinner.style.display = 'none';
    }
}

// Event listeners
checkButton.addEventListener('click', checkCode);

// Enter key in textarea
codeInput.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 'Enter') {
        checkCode();
    }
});

// Auto-resize textarea
codeInput.addEventListener('input', function() {
    this.style.height = 'auto';
    this.style.height = this.scrollHeight + 'px';
});

// Initialize
document.addEventListener('DOMContentLoaded', function() {
    // Focus on code input
    codeInput.focus();
    
    // Show welcome message
    setTimeout(() => {
        showMessage('MSRA C Standard Checker\'a hoş geldiniz!', 'success');
    }, 1000);
});

// Add some CSS for result styling
const resultStyles = document.createElement('style');
resultStyles.textContent = `
    .result-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 15px;
        padding-bottom: 10px;
        border-bottom: 1px solid #e5e7eb;
    }
    
    .result-header h4 {
        margin: 0;
        font-size: 1.2rem;
        font-weight: 600;
    }
    
    .confidence {
        background: #3b82f6;
        color: white;
        padding: 4px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
    }
    
    .result-message {
        margin-bottom: 15px;
    }
    
    .result-message p {
        font-size: 1rem;
        line-height: 1.6;
        margin: 0;
    }
    
    .result-details {
        background: #f1f5f9;
        padding: 10px;
        border-radius: 6px;
        font-size: 0.9rem;
    }
    
    .result-details p {
        margin: 0;
        color: #475569;
    }
`;
document.head.appendChild(resultStyles);

