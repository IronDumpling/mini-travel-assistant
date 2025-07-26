import React, { useMemo } from 'react';
import { Calendar, Clock, MapPin, Plane, Hotel, Camera, Star } from 'lucide-react';
import { format, addHours, startOfDay, endOfDay } from 'date-fns';
import { useTravelPlans } from '../../hooks/useApi';
import type { TravelPlan, CalendarEvent } from '../../types/api';

interface TravelCalendarProps {
  sessionId: string | null;
}

interface TimeSlotProps {
  hour: number;
  events: CalendarEvent[];
}

const TimeSlot: React.FC<TimeSlotProps> = ({ hour, events }) => {
  const formatHour = (h: number) => {
    if (h === 0) return '12:00 AM';
    if (h < 12) return `${h}:00 AM`;
    if (h === 12) return '12:00 PM';
    return `${h - 12}:00 PM`;
  };

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'flight': return <Plane className="w-3 h-3" />;
      case 'hotel': return <Hotel className="w-3 h-3" />;
      case 'attraction': return <Camera className="w-3 h-3" />;
      default: return <MapPin className="w-3 h-3" />;
    }
  };

  const getEventColor = (type: string) => {
    switch (type) {
      case 'flight': return 'bg-blue-100 text-blue-800 border-blue-200';
      case 'hotel': return 'bg-green-100 text-green-800 border-green-200';
      case 'attraction': return 'bg-purple-100 text-purple-800 border-purple-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  return (
    <div className="flex border-b border-gray-100">
      <div className="w-20 py-3 px-2 text-xs text-gray-500 font-medium border-r border-gray-100">
        {formatHour(hour)}
      </div>
      <div className="flex-1 p-2 min-h-[60px]">
        {events.map((event) => (
          <div
            key={event.id}
            className={`mb-1 p-2 rounded border text-xs ${getEventColor(event.type)}`}
          >
            <div className="flex items-center gap-1 mb-1">
              {getEventIcon(event.type)}
              <span className="font-medium truncate">{event.title}</span>
            </div>
            {event.description && (
              <div className="text-xs opacity-75 truncate">{event.description}</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export const TravelCalendar: React.FC<TravelCalendarProps> = ({ sessionId }) => {
  const { data: travelPlans, isLoading } = useTravelPlans(sessionId);

  const calendarEvents = useMemo(() => {
    if (!travelPlans || travelPlans.length === 0) return [];

    const events: CalendarEvent[] = [];
    const today = new Date();

    travelPlans.forEach((plan: TravelPlan) => {
      // Generate sample events from travel plan data
      
      // Add flights
      plan.flights.forEach((flight, index) => {
        const departureTime = new Date(flight.departure_time);
        const arrivalTime = new Date(flight.arrival_time);
        
        events.push({
          id: `flight-${plan.id}-${index}`,
          title: `${flight.airline} Flight`,
          start: departureTime,
          end: arrivalTime,
          description: `${flight.departure_time} - ${flight.arrival_time}`,
          type: 'flight',
          details: flight,
        });
      });

      // Add hotels (assume check-in at 3 PM, check-out at 11 AM)
      plan.hotels.forEach((hotel, index) => {
        const checkIn = addHours(startOfDay(today), 15); // 3 PM
        const checkOut = addHours(startOfDay(today), 11); // 11 AM next day
        
        events.push({
          id: `hotel-${plan.id}-${index}`,
          title: hotel.name,
          start: checkIn,
          end: checkOut,
          description: `$${hotel.price_per_night}/night`,
          type: 'hotel',
          details: hotel,
        });
      });

      // Add attractions (assume 2-hour visits starting at various times)
      plan.attractions.forEach((attraction, index) => {
        const visitTime = addHours(startOfDay(today), 9 + (index * 3)); // Starting at 9 AM, 3 hours apart
        
        events.push({
          id: `attraction-${plan.id}-${index}`,
          title: attraction.name,
          start: visitTime,
          end: addHours(visitTime, 2),
          description: attraction.category,
          type: 'attraction',
          details: attraction,
        });
      });
    });

    return events;
  }, [travelPlans]);

  const hourlyEvents = useMemo(() => {
    const eventsByHour: { [hour: number]: CalendarEvent[] } = {};
    
    for (let hour = 0; hour < 24; hour++) {
      eventsByHour[hour] = [];
    }

    calendarEvents.forEach((event) => {
      const startHour = event.start.getHours();
      eventsByHour[startHour].push(event);
    });

    return eventsByHour;
  }, [calendarEvents]);

  if (!sessionId) {
    return (
      <div className="w-80 bg-gray-50 border-l border-gray-200 flex items-center justify-center">
        <div className="text-center text-gray-500 p-6">
          <Calendar className="w-12 h-12 mx-auto mb-4 opacity-50" />
          <h3 className="font-medium mb-2">Travel Calendar</h3>
          <p className="text-sm">Select a session to view travel plans</p>
        </div>
      </div>
    );
  }

  return (
    <div className="w-80 bg-white border-l border-gray-200 flex flex-col h-full">
      {/* Header */}
      <div className="flex-shrink-0 p-4 border-b border-gray-200">
        <div className="flex items-center gap-2 mb-3">
          <Calendar className="w-5 h-5 text-blue-600" />
          <h2 className="font-semibold text-gray-900">Travel Schedule</h2>
        </div>
        
        <div className="text-sm text-gray-600">
          {format(new Date(), 'EEEE, MMMM d, yyyy')}
        </div>
        
        {/* Legend */}
        <div className="mt-3 space-y-1">
          <div className="flex items-center gap-2 text-xs">
            <div className="flex items-center gap-1">
              <Plane className="w-3 h-3 text-blue-600" />
              <span>Flights</span>
            </div>
            <div className="flex items-center gap-1">
              <Hotel className="w-3 h-3 text-green-600" />
              <span>Hotels</span>
            </div>
            <div className="flex items-center gap-1">
              <Camera className="w-3 h-3 text-purple-600" />
              <span>Attractions</span>
            </div>
          </div>
        </div>
      </div>

      {/* Calendar Body */}
      <div className="flex-1 overflow-y-auto">
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
              <Calendar className="w-12 h-12 mx-auto mb-4 opacity-50" />
              <h3 className="font-medium mb-2">No Plans Yet</h3>
              <p className="text-sm">
                Start chatting to create your travel itinerary
              </p>
            </div>
          </div>
        ) : (
          <div className="divide-y divide-gray-100">
            {Array.from({ length: 24 }, (_, hour) => (
              <TimeSlot
                key={hour}
                hour={hour}
                events={hourlyEvents[hour] || []}
              />
            ))}
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