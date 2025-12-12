# Student Success Prediction - Frontend

This is the React + Vite frontend for the Student Success Prediction system.

## Features

- **Student Prediction Form**: Enter student academic and demographic data
- **Real-time Prediction Results**: Visual display of persistence predictions with confidence levels
- **Admin Dashboard**: View recent predictions with full details
- **Responsive Design**: Works on desktop, tablet, and mobile devices

## Tech Stack

- **React 18**: UI framework
- **Vite**: Build tool and dev server
- **CSS3**: Styling with modern features
- **Fetch API**: HTTP client for backend communication

## Project Structure

```
frontend/
├── src/
│   ├── api/
│   │   └── client.js          # API client functions
│   ├── components/
│   │   ├── StudentForm.jsx    # Input form component
│   │   ├── StudentForm.css
│   │   ├── PredictionResult.jsx  # Results display
│   │   ├── PredictionResult.css
│   │   ├── AdminPanel.jsx     # Recent predictions table
│   │   └── AdminPanel.css
│   ├── App.jsx                # Main app component
│   ├── App.css
│   ├── main.jsx               # Entry point
│   └── index.css              # Global styles
├── package.json
└── vite.config.js
```

## Development

### Install dependencies
```bash
npm install
```

### Run development server
```bash
npm run dev
```

The app will be available at `http://localhost:5173`

### Build for production
```bash
npm run build
```

The optimized build will be in the `dist/` folder.

## API Configuration

The frontend connects to the backend API at `http://localhost:8000` by default.

To change the API URL, edit `src/api/client.js`:

```javascript
const API_BASE_URL = 'http://your-api-url:port';
```

## Component Overview

### StudentForm
- Dynamically loads categorical field options from the API
- Validates required numeric fields
- Provides user-friendly labels for categorical values
- Handles form submission and error states

### PredictionResult
- Displays prediction outcome (persist vs. at-risk)
- Shows probability bars with animations
- Includes appropriate disclaimers
- Adapts styling based on prediction result

### AdminPanel
- Fetches and displays recent predictions
- Collapsible panel to save space
- Sortable table with key student data
- Color-coded prediction badges

## Design Principles

1. **Simplicity**: Clean, uncluttered interface
2. **Clarity**: Clear visual hierarchy and labels
3. **Responsiveness**: Mobile-first design approach
4. **Accessibility**: Semantic HTML and proper contrast ratios
5. **Performance**: Minimal dependencies, optimized rendering

## Browser Support

- Chrome (latest)
- Firefox (latest)
- Safari (latest)
- Edge (latest)

## License

Part of the COMP258 Group Project - Developing Full-Stack Intelligent Apps
