// API Types based on backend schemas

export interface ChatMessage {
  message: string;
  session_id?: string;
  enable_refinement?: boolean;
}

export interface ChatResponse {
  success: boolean;
  content: string;
  confidence: number;
  actions_taken: string[];
  next_steps: string[];
  session_id: string;
  refinement_details?: any;
  plan_changes?: {
    success: boolean;
    changes_made: string[];
    events_added: number;
    events_updated: number;
    events_deleted: number;
    metadata_updated: boolean;
    plan_modifications?: {
      reason: string;
      impact: string;
    };
    error?: string;
  };
}

export interface Session {
  session_id: string;
  title: string;
  description?: string;
  created_at: string;
  updated_at: string;
  is_active: boolean;
}

export interface SessionCreate {
  title?: string;
  description?: string;
}

export interface SessionResponse {
  session_id: string;
  message: string;
  session?: Session;
}

export interface SessionsListResponse {
  sessions: Session[];
  current_session?: string;
  total: number;
}

export enum TripStyle {
  RELAXED = "relaxed",
  ADVENTURE = "adventure", 
  CULTURAL = "cultural",
  LUXURY = "luxury",
  BUDGET = "budget"
}

export interface TravelRequest {
  destination: string;
  origin?: string;
  duration_days?: number;
  travelers: number;
  budget?: number;
  budget_currency: string;
  trip_style: TripStyle;
  interests: string[];
  additional_requirements?: string;
}

export interface Attraction {
  name: string;
  rating: number;
  description: string;
  location: string;
  category: string;
  estimated_cost: number;
  photos: string[];
}

export interface Hotel {
  name: string;
  rating: number;
  price_per_night: number;
  location: string;
  amenities: string[];
}

export interface Flight {
  airline: string;
  price: number;
  duration: number; // minutes
  departure_time: string;
  arrival_time: string;
}

export interface FrameworkMetadata {
  confidence: number;
  actions_taken: string[];
  next_steps: string[];
  tools_used: string[];
  processing_time: number;
  intent_analysis?: any;
  quality_score?: number;
  refinement_iterations?: number;
}

export interface TravelPlan {
  id: string;
  request: TravelRequest;
  content: string;
  attractions: Attraction[];
  hotels: Hotel[];
  flights: Flight[];
  metadata: FrameworkMetadata;
  created_at: string;
  session_id?: string;
  status: string;
}

export interface TravelPlanResponse {
  success: boolean;
  plan?: TravelPlan;
  error?: string;
  message?: string;
}

export interface ChatHistory {
  conversation_id: string;
  messages: Array<{
    role: 'user' | 'assistant';
    content: string;
    timestamp: string;
    metadata?: any;
  }>;
}

// Calendar Event for the right panel
export interface CalendarEvent {
  id: string;
  title: string;
  start: Date;
  end: Date;
  description?: string;
  type: 'flight' | 'hotel' | 'attraction' | 'activity' | 'restaurant' | 'transportation' | 'meeting' | 'free_time';
  details?: any;
  location?: string;
  confidence?: number;
  source?: string;
}

// Session Travel Plan types
export interface TravelPlanMetadata {
  destination?: string;
  duration_days?: number;
  travelers?: number;
  budget?: number;
  budget_currency?: string;
  interests?: string[];
  last_updated?: string;
  confidence?: number;
  completion_status?: string;
}

export interface SessionTravelPlan {
  plan_id: string;
  session_id: string;
  events: CalendarEvent[];
  metadata: TravelPlanMetadata;
  created_at: string;
  updated_at: string;
}

export interface PlanResponse {
  success: boolean;
  plan?: SessionTravelPlan;
  message: string;
  events_count: number;
} 