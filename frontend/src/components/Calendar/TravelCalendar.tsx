import React, { useMemo } from 'react';
import { Calendar as CalendarIcon, Clock, MapPin, Plane, Hotel, Camera, Star } from 'lucide-react';
import { Calendar, dateFnsLocalizer, Views, View } from 'react-big-calendar';
import { format, parse, startOfWeek, getDay, addDays, startOfDay, addHours } from 'date-fns';
import { enUS } from 'date-fns/locale';
import { useTravelPlans } from '../../hooks/useApi';
import type { TravelPlan, CalendarEvent, SessionTravelPlan } from '../../types/api';
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
      default: return <MapPin className="w-3 h-3" />;
    }
  };

  return (
    <div className="flex items-center gap-1 text-xs">
      {getEventIcon(event.type)}
      <span className="truncate">{event.title}</span>
    </div>
  );
};

export const TravelCalendar: React.FC<TravelCalendarProps> = ({ sessionId }) => {
  const { data: travelPlans, isLoading } = useTravelPlans(sessionId);

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
            
            const startDate = new Date(startField);
            const endDate = new Date(endField);
            
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
            
            return {
              id: event.id,
              title: event.title,
              start: startDate,
              end: endDate,
              type: (event as any).event_type || (event as any).type,
              location: event.location,
              description: event.description,
              resource: event
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
          const departureTime = new Date(flight.departure_time);
          const arrivalTime = new Date(flight.arrival_time);
          
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

        // Add hotels (assume check-in at 3 PM, check-out at 11 AM)
        plan.hotels?.forEach((hotel: any, index: number) => {
          const checkIn = addHours(startOfDay(today), 15); // 3 PM
          const checkOut = addHours(startOfDay(addDays(today, 1)), 11); // 11 AM next day
          
          events.push({
            id: `hotel-${plan.id}-${index}`,
            title: hotel.name,
            start: checkIn,
            end: checkOut,
            type: 'hotel',
            location: hotel.location,
            description: `$${hotel.price_per_night}/night`,
          });
        });

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

  // Custom event style getter
  const eventStyleGetter = (event: any) => {
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
        backgroundColor = '#ea580c'; // orange
        break;
      default:
        backgroundColor = '#6b7280'; // gray
    }

    return {
      style: {
        backgroundColor,
        color,
        border: 'none',
        borderRadius: '4px',
        fontSize: '12px',
        padding: '2px 4px'
      }
    };
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

  return (
    <div className="w-[1000px] bg-white border-l border-gray-200 flex flex-col h-full">
      {/* Header */}
      <div className="flex-shrink-0 p-4 border-b border-gray-200">
        <div className="flex items-center gap-2 mb-3">
          <CalendarIcon className="w-5 h-5 text-blue-600" />
          <h2 className="font-semibold text-gray-900">Travel Schedule</h2>
        </div>
        
        <div className="text-sm text-gray-600">
          {format(new Date(), 'EEEE, MMMM d, yyyy')}
        </div>
        
        {/* Legend */}
        <div className="mt-3 space-y-1">
          <div className="flex items-center gap-2 text-xs">
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
              <CalendarIcon className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <h3 className="font-medium mb-2">No Plans Yet</h3>
              <p className="text-sm">
                Start chatting to create your travel itinerary
              </p>
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