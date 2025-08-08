# Mini Travel Assistant Frontend

A modern React + TypeScript frontend for the AI-powered travel planning assistant. Features a ChatGPT-style interface with three main panels: session management, chat interface, and travel calendar.

## 🏗️ Architecture

### Three-Panel Layout
- **Left Panel**: Session management sidebar for creating, switching, and deleting travel planning sessions
- **Middle Panel**: Chat interface with conversation history and message input
- **Right Panel**: 24-hour calendar view displaying travel plans and itineraries

### Technology Stack
- **Framework**: React 18 + TypeScript
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **State Management**: React Query (@tanstack/react-query)
- **HTTP Client**: Axios
- **Icons**: Lucide React
- **Date Handling**: date-fns

## 🚀 Getting Started

### Prerequisites
- Node.js 18+
- npm or yarn
- Backend API server running on http://localhost:8000

### Installation

1. **Navigate to frontend directory**:
   ```bash
   cd frontend
   ```

2. **Install dependencies**:
   ```bash
   npm install
   ```

3. **Start development server**:
   ```bash
   npm run dev
   ```

4. **Open in browser**:
   Navigate to http://localhost:3000

### Build for Production

```bash
npm run build
npm run preview
```

## 🎯 Features

### Session Management
- ✅ Create new travel planning sessions
- ✅ Switch between existing sessions
- ✅ Delete sessions with confirmation
- ✅ Auto-select most recent session
- ✅ Real-time session list updates

### Chat Interface
- ✅ Send messages to AI travel agent
- ✅ View conversation history
- ✅ Real-time message updates
- ✅ Loading states and error handling
- ✅ Confidence scoring display
- ✅ Auto-scroll to latest messages

### Travel Calendar
- ✅ 24-hour day view format
- ✅ Dynamic event generation from travel plans
- ✅ Categorized events (flights, hotels, attractions)
- ✅ Color-coded event types
- ✅ Read-only calendar display
- ✅ Real-time plan updates

## 🔧 Configuration

### API Proxy
The Vite development server is configured to proxy API requests to the backend:

```typescript
// vite.config.ts
server: {
  proxy: {
    '/api': {
      target: 'http://localhost:8000',
      changeOrigin: true,
    },
  },
}
```

### Environment Variables
Create a `.env` file in the frontend directory for custom configuration:

```env
VITE_API_BASE_URL=http://localhost:8000
VITE_APP_TITLE=Mini Travel Assistant
```

## 📁 Project Structure

```
frontend/
├── public/                     # Static assets
├── src/
│   ├── components/             # React components
│   │   ├── Chat/              # Chat interface components
│   │   │   └── ChatInterface.tsx
│   │   ├── Sidebar/           # Session management components
│   │   │   └── SessionSidebar.tsx
│   │   └── Calendar/          # Travel calendar components
│   │       └── TravelCalendar.tsx
│   ├── hooks/                 # Custom React hooks
│   │   └── useApi.ts         # API state management hooks
│   ├── services/              # API services
│   │   └── api.ts            # Axios API client
│   ├── types/                 # TypeScript type definitions
│   │   └── api.ts            # API response types
│   ├── utils/                 # Utility functions
│   ├── App.tsx               # Main application component
│   ├── main.tsx              # Application entry point
│   └── index.css             # Global styles
├── index.html                  # HTML entry point
├── package.json               # Dependencies and scripts
├── tsconfig.json              # TypeScript configuration
├── vite.config.ts             # Vite configuration
├── tailwind.config.js         # Tailwind CSS configuration
└── postcss.config.js          # PostCSS configuration
```

## 🔄 API Integration

### Supported Endpoints
- `GET /api/sessions` - List sessions
- `POST /api/sessions` - Create session
- `PUT /api/sessions/{id}/switch` - Switch session
- `DELETE /api/sessions/{id}` - Delete session
- `POST /api/chat` - Send chat message
- `GET /api/chat/history/{sessionId}` - Get chat history
- `GET /api/plans?session_id={sessionId}` - Get travel plans

### React Query Configuration
The app uses React Query for efficient API state management:

```typescript
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 1000 * 60 * 5, // 5 minutes
      retry: 1,
    },
  },
});
```

## 🎨 UI Components

### Design System
- **Colors**: Blue primary, gray neutrals
- **Typography**: System font stack
- **Icons**: Lucide React icon library
- **Layout**: Flexbox with Tailwind CSS

### Responsive Design
- **Desktop First**: Optimized for desktop use (minimum 1200px width)
- **Fixed Layout**: Three-panel layout with fixed panel widths
- **Scrollable Areas**: Chat history and calendar events scroll independently

## 🔍 State Management

### Session State
```typescript
const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
```

### API State (React Query)
- `useSessions()` - Session list and current session
- `useChatHistory(sessionId)` - Chat conversation history
- `useTravelPlans(sessionId)` - Travel plans for calendar
- `useSendMessage()` - Send chat message mutation

## 🎯 User Experience

### Loading States
- Skeleton loading for initial data
- Spinners for API operations
- Disabled states during processing

### Error Handling
- Network error messages
- Retry capabilities
- Graceful degradation

### Real-time Updates
- Automatic cache invalidation
- Optimistic updates
- Live data synchronization

## 📊 Performance

### Optimization Features
- React Query caching and background updates
- Component lazy loading where appropriate
- Efficient re-renders with proper dependency arrays
- Memoized expensive calculations

### Bundle Size
- Tree-shaking enabled
- Production builds optimized with Vite
- Dynamic imports for code splitting

## 🧪 Development

### Available Scripts
- `npm run dev` - Start development server
- `npm run build` - Build for production
- `npm run preview` - Preview production build
- `npm run lint` - Run ESLint

### Code Style
- TypeScript strict mode
- ESLint for code quality
- Consistent component patterns
- Functional components with hooks

## 🚀 Deployment

### Build Output
```bash
npm run build
```
Generates optimized static files in `dist/` directory.

### Deployment Options
- **Static Hosting**: Netlify, Vercel, GitHub Pages
- **CDN**: AWS S3 + CloudFront
- **Server**: nginx, Apache

### Environment Configuration
Ensure the backend API URL is correctly configured for production deployment.

## 🔒 Security

### API Security
- CORS configured on backend
- No sensitive data in frontend code
- Secure HTTP-only cookies for authentication (when implemented)

### Content Security
- No dangerous innerHTML usage
- XSS protection via React's built-in escaping
- Secure handling of user input

## 📈 Future Enhancements

### Planned Features
- 🔄 Real-time notifications
- 📱 Mobile responsive design
- 🌙 Dark mode support
- 🔐 User authentication
- 📷 Image upload for travel plans
- 🗺️ Interactive maps integration
- 📱 Progressive Web App (PWA) capabilities

### Technical Improvements
- 🧪 Unit and integration tests
- 🎨 Storybook component library
- 📊 Performance monitoring
- 🔄 Offline support
- 📦 Bundle size optimization

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the backend API documentation
- Verify API server is running
- Check browser console for errors
- Review network requests in dev tools

---

**Note**: This frontend is designed to work with the Mini Travel Assistant backend API. Ensure the backend server is running and accessible for full functionality. 