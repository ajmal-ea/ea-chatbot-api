// Configuration for Express Analytics Chatbot
const EA_CHATBOT_CONFIG = {
    // API endpoint URL - Automatically detect environment
    apiBaseUrl: detectApiUrl(),
    
    // Chatbot appearance
    chatbotTitle: 'Express Analytics AI Assistant',
    logoUrl: 'https://d3caycb064h6u1.cloudfront.net/wp-content/uploads/2024/08/Logo.svg',
    
    // Welcome message
    welcomeMessage: `👋 Welcome to Express Analytics! I'm your AI assistant, ready to help you with:
    <ul>
        <li>Data Analytics inquiries</li>
        <li>Marketing Analytics questions</li>
        <li>AI and Machine Learning solutions</li>
        <li>Business Intelligence insights</li>
    </ul>
    How can I assist you today?`,
    
    // Meeting scheduler
    meetingSchedulerUrl: 'https://calendly.com/expressanalytics/30min',
    
    // Branding
    poweredByText: 'Powered by Express Analytics',
    poweredByLink: 'https://www.expressanalytics.com'
};

// Function to detect the appropriate API URL based on environment
function detectApiUrl() {
    // Check if we're running in Docker by looking at the hostname
    const isDocker = window.location.hostname === 'localhost' && 
                     (window.location.port === '4000' || window.location.port === '');
    
    if (isDocker) {
        // When running in Docker, use the service name from docker-compose
        return 'http://chatbot-api:8000';
    } else if (window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1') {
        // Local development outside Docker
        return 'http://localhost:8000';
    } else {
        // Production environment - replace with your actual production API URL
        return 'https://api.expressanalytics.com';
    }
} 