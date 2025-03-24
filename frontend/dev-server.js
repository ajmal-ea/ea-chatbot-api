const express = require('express');
const cors = require('cors');
const path = require('path');
const app = express();
const port = 4000;

// Enable CORS for all routes
app.use(cors());

// Serve static files from the 'static' directory
app.use(express.static(path.join(__dirname, 'static')));

// Serve the test-embed.html file at the root
app.get('/', (req, res) => {
  res.sendFile(path.join(__dirname, 'test-embed.html'));
});

// Determine the host to listen on (0.0.0.0 for Docker, localhost for local dev)
// Check if we're in a production environment
const isProduction = process.env.NODE_ENV === 'production';
const host = isProduction ? '0.0.0.0' : 'localhost';

// Start the server
app.listen(port, host, () => {
  console.log(`Express Analytics Chatbot dev server running at http://${host === '0.0.0.0' ? 'localhost' : host}:${port}`);
  console.log(`Server is listening on ${host}:${port}`);
  console.log(`Environment: ${isProduction ? 'Production' : 'Development'}`);
  console.log(`Make sure your API server is running at http://localhost:8000`);
}); 