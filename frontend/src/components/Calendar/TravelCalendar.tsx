import React, { useMemo } from 'react';
import { Calendar as CalendarIcon, Clock, MapPin, Plane, Hotel, Camera, Star, Loader2 } from 'lucide-react';
import { Calendar, dateFnsLocalizer, Views } from 'react-big-calendar';
import { format, parse, startOfWeek, getDay, addDays, startOfDay, addHours } from 'date-fns';
import { enUS } from 'date-fns/locale';
import { useTravelPlans, usePlanGenerationStatus } from '../../hooks/useApi';
import type { TravelPlan, SessionTravelPlan } from '../../types/api';
import 'react-big-calendar/lib/css/react-big-calendar.css';

interface TravelCalendarProps {
  sessionId: string | null;
}

// Setup the localizer for react-big-calendar
const localizer = dateFnsLocalizer({
  format,
  parse,
  startOfWeek,
  getDay,
  locales: {
    'en-US': enUS,
  },
});

// Custom event component to show event details
const EventComponent = ({ event }: { event: any }) => {
  const getEventIcon = (type: string) => {
    switch (type) {
      case 'flight': return <Plane className="w-3 h-3" />;
      case 'hotel': return <Hotel className="w-3 h-3" />;
      case 'attraction': return <Camera className="w-3 h-3" />;
      case 'restaurant':
      case 'meal': return <Star className="w-3 h-3" />; // Star icon for food-related events
      case 'transportation': return <Plane className="w-3 h-3" />; // Reuse plane for transport
      case 'activity': return <Camera className="w-3 h-3" />; // Similar to attractions
      case 'meeting': return <Clock className="w-3 h-3" />;
      case 'free_time': return <Clock className="w-3 h-3" />;
      default: return <MapPin className="w-3 h-3" />;
    }
  };

  // ✅ Enhanced flight event display with airline and flight details
  const renderFlightDetails = () => {
    const details = event.resource?.details || {};
    const airline = details.airline || 'TBA';
    const flightNumber = details.flight_number || '';
    const price = details.price?.amount || 'TBA';
    const currency = details.price?.currency || 'USD';
    const durationMinutes = details.duration_minutes || null;
    
    // Extract route from location or title
    const route = event.location || '';
    
    // Build compact flight info
    const parts = [];
    
    // Add airline and flight number
    if (airline !== 'TBA' && flightNumber && flightNumber !== 'TBA') {
      parts.push(`${airline} ${flightNumber}`);
    } else if (airline !== 'TBA') {
      parts.push(airline);
    }
    
    // Add route if available
    if (route) {
      parts.push(route);
    }
    
    // Add duration if available
    if (durationMinutes && durationMinutes > 0) {
      const hours = Math.floor(durationMinutes / 60);
      const minutes = durationMinutes % 60;
      if (hours > 0 && minutes > 0) {
        parts.push(`${hours}h${minutes}m`);
      } else if (hours > 0) {
        parts.push(`${hours}h`);
      } else if (minutes > 0) {
        parts.push(`${minutes}m`);
      }
    }
    
    // Add price if available
    if (price !== 'TBA' && price !== undefined && price !== null) {
      parts.push(`${price} ${currency}`);
    }
    
    return parts.length > 0 ? parts.join(' • ') : event.title;
  };

  // ✅ Enhanced hotel event display
  const renderHotelDetails = () => {
    const details = event.resource?.details || {};
    const rating = details.rating;
    const nights = details.nights;
    const pricePerNight = details.price_per_night?.amount;
    const currency = details.price_per_night?.currency || 'USD';
    
    const parts = [event.title];
    
    if (rating) {
      parts.push(`★${rating}`);
    }
    
    if (nights) {
      parts.push(`${nights}n`);
    }
    
    if (pricePerNight) {
      parts.push(`${pricePerNight}${currency}/n`);
    }
    
    return parts.join(' • ');
  };

  // ✅ Smart display title based on event type
  const getDisplayContent = () => {
    const isAllDay = event.allDay;
    
    if (event.type === 'flight') {
      const flightInfo = renderFlightDetails();
      return isAllDay ? `All Day - ${flightInfo}` : flightInfo;
    }
    
    if (event.type === 'hotel') {
      const hotelInfo = renderHotelDetails();
      return isAllDay ? `All Day - ${hotelInfo}` : hotelInfo;
    }
    
    // Default display for other event types
    const defaultTitle = event.title;
    return isAllDay ? `All Day - ${defaultTitle}` : defaultTitle;
  };

  return (
    <div className="flex items-center gap-1 text-xs leading-tight">
      {getEventIcon(event.type)}
      <span className="flex-1 whitespace-normal break-words text-wrap leading-tight">
        {getDisplayContent()}
      </span>
    </div>
  );
};

// Utility function for safe datetime parsing
const parseDateTime = (dateString: string): Date => {
  // Standardize datetime format handling
  let normalizedDate = dateString;
  
  // If space-separated format, convert to standard ISO format
  if (dateString.includes(' ') && !dateString.includes('T')) {
    normalizedDate = dateString.replace(' ', 'T');
  }
  
  // Ensure timezone information is present
  if (!normalizedDate.includes('+') && !normalizedDate.includes('Z')) {
    normalizedDate += '+00:00';
  }
  
  const date = new Date(normalizedDate);
  
  // Validate date validity
  if (isNaN(date.getTime())) {
    console.error('Invalid date format:', dateString, 'using current time as fallback');
    return new Date(); // Return current time as fallback
  }
  
  return date;
};

export const TravelCalendar: React.FC<TravelCalendarProps> = ({ sessionId }) => {
  const { data: travelPlans, isLoading } = useTravelPlans(sessionId);
  const { data: planStatus } = usePlanGenerationStatus(sessionId);

  // Add custom styles for hover effects
  const customStyles = `
    .rbc-calendar .rbc-event:hover {
      cursor: pointer !important;
    }
    
    .rbc-calendar .rbc-event.travel-calendar-event[style*="background-color: rgb(37, 99, 235)"]:hover {
      background-color: #1d4ed8 !important; /* darker blue for flights */
    }
    
    .rbc-calendar .rbc-event.travel-calendar-event[style*="background-color: rgb(22, 163, 74)"]:hover {
      background-color: #15803d !important; /* darker green for hotels */
    }
    
    .rbc-calendar .rbc-event.travel-calendar-event[style*="background-color: rgb(147, 51, 234)"]:hover {
      background-color: #7c3aed !important; /* darker purple for attractions */
    }
    
    .rbc-calendar .rbc-event.travel-calendar-event[style*="background-color: rgb(234, 88, 12)"]:hover {
      background-color: #dc2626 !important; /* red for restaurants */
    }
    
    .rbc-calendar .rbc-event.travel-calendar-event[style*="background-color: rgb(107, 114, 128)"]:hover {
      background-color: #4b5563 !important; /* darker gray for default */
    }
    
    /* ✅ Ultra-strong override for all selection states */
    .rbc-calendar .rbc-event.rbc-selected,
    .rbc-calendar .rbc-event.rbc-selected:hover,
    .rbc-calendar .rbc-event.rbc-selected:focus,
    .rbc-calendar .rbc-event.rbc-selected:active {
      background-color: inherit !important;
      color: white !important;
      border: inherit !important;
      outline: none !important;
      box-shadow: none !important;
      opacity: 1 !important;
      filter: none !important;
    }
    
    .rbc-calendar .rbc-event:focus {
      outline: none !important;
      box-shadow: none !important;
      opacity: 1 !important;
      color: white !important;
    }
    
    /* ✅ Specific override for travel calendar events */
    .rbc-calendar .rbc-event.travel-calendar-event.rbc-selected,
    .rbc-calendar .rbc-event.travel-calendar-event.rbc-selected:hover,
    .rbc-calendar .rbc-event.travel-calendar-event.rbc-selected:focus,
    .rbc-calendar .rbc-event.travel-calendar-event.rbc-selected:active {
      background-color: inherit !important;
      color: white !important;
      border: inherit !important;
      border-radius: inherit !important;
      opacity: 1 !important;
      filter: none !important;
    }
    
    /* ✅ Override any potential inline styles from react-big-calendar */
    .rbc-calendar .rbc-event[style] {
      opacity: 1 !important;
    }
    
    .rbc-calendar .rbc-event.rbc-selected[style] {
      opacity: 1 !important;
      filter: none !important;
    }
    
    /* ✅ Disable any selection overlay effects */
    .rbc-calendar .rbc-selected-overlay {
      display: none !important;
    }
    
    /* ✅ Ensure active/clicked states don't change appearance */
    .rbc-calendar .rbc-event:active,
    .rbc-calendar .rbc-event:active:focus {
      background-color: inherit !important;
      color: white !important;
      border: inherit !important;
      outline: none !important;
      box-shadow: none !important;
      opacity: 1 !important;
      filter: none !important;
    }
    
    /* ✅ Fix event title wrapping and overlapping layout */
    .rbc-calendar .rbc-event {
      white-space: normal !important; /* Allow text wrapping */
      word-wrap: break-word !important;
      overflow: visible !important;
      line-height: 1.2 !important;
      min-height: 20px !important;
      text-overflow: clip !important; /* Remove ellipsis */
    }
    
    .rbc-calendar .rbc-event-content {
      white-space: normal !important;
      overflow: visible !important;
      text-overflow: clip !important;
      word-break: break-word !important;
      line-height: 1.2 !important;
    }
    
    /* ✅ Improve overlapping events layout */
    .rbc-calendar .rbc-row-segment {
      z-index: auto !important;
    }
    
    .rbc-calendar .rbc-events-container {
      margin-right: 0 !important;
    }
    
    /* ✅ Better spacing for concurrent events */
    .rbc-calendar .rbc-event + .rbc-event {
      margin-left: 1px !important;
    }
    
    /* ✅ All-day events styling - applies to all event types */
    .rbc-calendar .rbc-allday-cell {
      min-height: 40px !important;
    }
    
    .rbc-calendar .rbc-event.rbc-all-day-event {
      margin-bottom: 2px !important;
      border-radius: 4px !important;
      font-size: 12px !important;
      padding: 4px 6px !important;
      min-height: 24px !important;
    }
    
    /* ✅ Ensure proper stacking of multiple all-day events */
    .rbc-calendar .rbc-row-segment.rbc-all-day-event {
      margin-bottom: 1px !important;
    }
    
    /* ✅ All-day event container */
    .rbc-calendar .rbc-row-content {
      overflow: visible !important;
    }
    
    .rbc-calendar .rbc-addons-dnd .rbc-addons-dnd-row-body {
      overflow: visible !important;
    }
  `;

  const calendarEvents = useMemo(() => {
    if (!travelPlans) return [];

    // If travelPlans is a SessionTravelPlan, use its events directly
    if (travelPlans && typeof travelPlans === 'object' && 'events' in travelPlans) {
      const sessionPlan = travelPlans as SessionTravelPlan;
      return sessionPlan.events
        .filter(event => {
          // Handle both backend (start_time/end_time) and frontend (start/end) field names
          const startField = (event as any).start_time || (event as any).start;
          const endField = (event as any).end_time || (event as any).end;
          
          if (!startField || !endField) {
            console.warn('Event missing start or end time:', event);
            return false;
          }
          return true;
        })
        .map(event => {
          try {
            // Handle both backend (start_time/end_time) and frontend (start/end) field names
            const startField = (event as any).start_time || (event as any).start;
            const endField = (event as any).end_time || (event as any).end;
            
            const startDate = parseDateTime(startField);
            const endDate = parseDateTime(endField);
            
            // Validate that dates are valid
            if (isNaN(startDate.getTime()) || isNaN(endDate.getTime())) {
              console.warn('Invalid dates for event:', event, {
                startField,
                endField,
                startDate,
                endDate
              });
              return null;
            }
            
            // Determine if this should be an all-day event
            const eventType = (event as any).event_type || (event as any).type;
            
            // Check multiple conditions for all-day events
            const isAllDayEvent = 
              // Hotel/accommodation events
              eventType === 'hotel' || 
              eventType === 'accommodation' ||
              (event.title && event.title.toLowerCase().includes('hotel')) ||
              
              // Meal/food budget events (typically all-day)
              eventType === 'meal' ||
              eventType === 'food' ||
              eventType === 'budget' ||
              (event.title && (
                event.title.toLowerCase().includes('budget') ||
                event.title.toLowerCase().includes('food') ||
                event.title.toLowerCase().includes('meal') ||
                event.title.toLowerCase().includes('dining')
              )) ||
              
              // Transportation (some may be all-day like train passes)
              (eventType === 'transportation' && 
               event.title && (
                 event.title.toLowerCase().includes('pass') ||
                 event.title.toLowerCase().includes('card') ||
                 event.title.toLowerCase().includes('metro')
               )) ||
              
              // General all-day activities
              eventType === 'all_day' ||
              eventType === 'full_day' ||
              
              // Events explicitly marked as all-day in backend
              (event as any).all_day === true ||
              (event as any).allDay === true;

            return {
              id: event.id,
              title: event.title,
              start: startDate,
              end: endDate,
              type: eventType,
              location: event.location,
              description: event.description,
              resource: event,
              allDay: isAllDayEvent // ✅ Mark hotel events as all-day
            };
          } catch (error) {
            console.warn('Error processing event dates:', event, error);
            return null;
          }
        })
        .filter(event => event !== null);
    }

    // Legacy support for old TravelPlan[] format
    const plansArray = travelPlans as TravelPlan[];
    if (Array.isArray(plansArray)) {
      const events: any[] = [];
      const today = new Date();

      plansArray.forEach((plan: TravelPlan) => {
        // Add flights
        plan.flights?.forEach((flight: any, index: number) => {
          const departureTime = parseDateTime(flight.departure_time);
          const arrivalTime = parseDateTime(flight.arrival_time);
          
          events.push({
            id: `flight-${plan.id}-${index}`,
            title: `${flight.airline} Flight`,
            start: departureTime,
            end: arrivalTime,
            type: 'flight',
            location: 'Airport',
            description: `${flight.departure_time} - ${flight.arrival_time}`,
          });
        });

        // Add hotels as all-day events
        plan.hotels?.forEach((hotel: any, index: number) => {
          const checkInDate = startOfDay(today);
          const checkOutDate = startOfDay(addDays(today, 1));
          
          events.push({
            id: `hotel-${plan.id}-${index}`,
            title: hotel.name,
            start: checkInDate,
            end: checkOutDate,
            type: 'hotel',
            location: hotel.location,
            description: `$${hotel.price_per_night}/night`,
            allDay: true // ✅ Mark as all-day event
          });
        });

        // Add meals/food budget as all-day events if they exist
        const planWithMeals = plan as any;
        if (planWithMeals.meals) {
          planWithMeals.meals.forEach((meal: any, index: number) => {
            const mealDate = startOfDay(today);
            
            events.push({
              id: `meal-${plan.id}-${index}`,
              title: meal.name || `Budget for Food`,
              start: mealDate,
              end: mealDate,
              type: 'meal',
              location: meal.location || planWithMeals.destination || 'Unknown',
              description: meal.description || `Daily food budget`,
              allDay: true // ✅ Mark meal budgets as all-day events
            });
          });
        }

        // Add attractions (assume 2-hour visits starting at various times)
        plan.attractions?.forEach((attraction: any, index: number) => {
          const visitTime = addHours(startOfDay(today), 9 + (index * 3)); // Starting at 9 AM, 3 hours apart
          
          events.push({
            id: `attraction-${plan.id}-${index}`,
            title: attraction.name,
            start: visitTime,
            end: addHours(visitTime, 2),
            type: 'attraction',
            location: attraction.location,
            description: attraction.category,
          });
        });
      });

      return events;
    }

    return [];
  }, [travelPlans]);

  // Custom event style getter with forced selection state handling
  const eventStyleGetter = (event: any, _start: Date, _end: Date, _isSelected: boolean) => {
    let backgroundColor = '#3174ad';
    let color = 'white';

    switch (event.type) {
      case 'flight':
        backgroundColor = '#2563eb'; // blue
        break;
      case 'hotel':
        backgroundColor = '#16a34a'; // green
        break;
      case 'attraction':
        backgroundColor = '#9333ea'; // purple
        break;
      case 'restaurant':
      case 'meal':
        backgroundColor = '#ea580c'; // orange for food-related events
        break;
      case 'transportation':
        backgroundColor = '#0ea5e9'; // sky blue
        break;
      case 'activity':
        backgroundColor = '#8b5cf6'; // violet
        break;
      case 'meeting':
        backgroundColor = '#ef4444'; // red
        break;
      case 'free_time':
        backgroundColor = '#10b981'; // emerald
        break;
      default:
        backgroundColor = '#6b7280'; // gray
    }

    // ✅ Force same styling regardless of selection state
    const baseStyle = {
      backgroundColor: backgroundColor, // ✅ Force original color even when selected
      color: color, // ✅ Force white text even when selected
      border: 'none',
      borderRadius: '4px',
      fontSize: '12px',
      cursor: 'pointer',
      transition: 'background-color 0.2s ease',
      outline: 'none',
      boxShadow: 'none',
      whiteSpace: 'normal' as 'normal',
      wordBreak: 'break-word' as 'break-word',
      lineHeight: '1.2',
      overflow: 'visible' as 'visible',
      textOverflow: 'clip' as 'clip',
      // ✅ Force these properties to override any selection styling
      opacity: '1',
      filter: 'none'
    };

    if (event.allDay) {
      // All-day event styling (applies to all event types marked as all-day)
      return {
        style: {
          ...baseStyle,
          padding: '4px 8px',
          minHeight: '24px',
          fontWeight: '500',
          // Slightly different styling for all-day events
          borderLeft: `4px solid ${backgroundColor}`,
          backgroundColor: `${backgroundColor}f0`, // Add transparency
          color: color, // ✅ Ensure text stays white
        },
        className: 'travel-calendar-event rbc-all-day-event'
      };
    } else {
      // Regular timed event styling
      return {
        style: {
          ...baseStyle,
          padding: '2px 4px',
          minHeight: '20px',
          backgroundColor: backgroundColor, // ✅ Force background color
          color: color, // ✅ Force text color
        },
        className: 'travel-calendar-event'
      };
    }
  };

  if (!sessionId) {
    return (
      <div className="w-[1000px] bg-gray-50 border-l border-gray-200 flex items-center justify-center">
        <div className="text-center text-gray-500 p-6">
          <CalendarIcon className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <h3 className="font-medium mb-2">Travel Calendar</h3>
          <p className="text-sm">Select a session to view travel plans</p>
        </div>
      </div>
    );
  }

  // Check if plan is being generated
  const isPlanGenerating = planStatus?.plan_generation_status === 'pending';
  const planGenerationFailed = planStatus?.plan_generation_status === 'failed';

  return (
    <div className="w-[1000px] bg-white border-l border-gray-200 flex flex-col h-full">
      <style>{customStyles}</style>
      
      {/* Header */}
      <div className="flex-shrink-0 p-4 border-b border-gray-200">
        <div className="flex items-center gap-2 mb-3">
          <CalendarIcon className="w-5 h-5 text-blue-600" />
          {isPlanGenerating && (
            <Loader2 className="w-4 h-4 text-blue-600 animate-spin" />
          )}
          <h2 className="font-semibold text-gray-900">Travel Schedule</h2>
        </div>
        
        <div className="text-sm text-gray-600">
          {format(new Date(), 'EEEE, MMMM d, yyyy')}
        </div>
        
        {/* Legend */}
        <div className="mt-3 space-y-1">
          <div className="flex items-center gap-2 text-xs flex-wrap">
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-blue-600 rounded"></div>
              <span>Flights</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-green-600 rounded"></div>
              <span>Hotels</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-purple-600 rounded"></div>
              <span>Attractions</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-orange-600 rounded"></div>
              <span>Meals</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-sky-500 rounded"></div>
              <span>Transport</span>
            </div>
            <div className="flex items-center gap-1">
              <div className="w-3 h-3 bg-violet-500 rounded"></div>
              <span>Activities</span>
            </div>
          </div>
        </div>
      </div>

      {/* Calendar Body */}
      <div className="flex-1 overflow-hidden">
        {isLoading ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-500">
              <Clock className="w-8 h-8 mx-auto mb-2 animate-spin" />
              <p>Loading schedule...</p>
            </div>
          </div>
        ) : calendarEvents.length === 0 ? (
          <div className="flex items-center justify-center h-full">
            <div className="text-center text-gray-500 p-6">
              {isPlanGenerating ? (
                <>
                  <Loader2 className="w-12 h-12 mx-auto mb-4 opacity-50 animate-spin" />
                  <h3 className="font-medium mb-2">Generating Travel Plan</h3>
                  <p className="text-sm">
                    Creating your itinerary based on the conversation...
                  </p>
                </>
              ) : planGenerationFailed ? (
                <>
                  <CalendarIcon className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <h3 className="font-medium mb-2 text-red-600">Plan Generation Failed</h3>
                  <p className="text-sm">
                    {planStatus?.plan_generation_error || 'Failed to generate travel plan'}
                  </p>
                </>
              ) : (
                <>
                  <CalendarIcon className="w-12 h-12 mx-auto mb-4 opacity-50" />
                  <h3 className="font-medium mb-2">No Plans Yet</h3>
                  <p className="text-sm">
                    Start chatting to create your travel itinerary
                  </p>
                </>
              )}
            </div>
          </div>
        ) : (
          <div className="h-full p-2">
            <Calendar
              localizer={localizer}
              events={calendarEvents}
              startAccessor="start"
              endAccessor="end"
              style={{ height: '100%' }}
              views={[Views.WEEK, Views.DAY, Views.AGENDA]}
              defaultView={Views.WEEK}
              step={60}
              showMultiDayTimes
              components={{
                event: EventComponent
              }}
              eventPropGetter={eventStyleGetter}
              popup
              popupOffset={{ x: 30, y: 20 }}
              onSelectEvent={() => {
                // ✅ Immediately clear any selection to prevent styling changes
                return false;
              }}
              selectable={false} // ✅ Completely disable selection
              onSelectSlot={() => {}} // ✅ Disable slot selection
              selected={[]} // ✅ Force empty selection array
            />
          </div>
        )}
      </div>

      {/* Footer */}
      {calendarEvents.length > 0 && (
        <div className="flex-shrink-0 p-4 border-t border-gray-200 bg-gray-50">
          <div className="text-xs text-gray-600">
            <div className="flex items-center gap-2">
              <Star className="w-3 h-3" />
              <span>{calendarEvents.length} scheduled activities</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}; 