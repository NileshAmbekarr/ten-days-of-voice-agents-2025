import { Button } from "@/components/livekit/button";
import React, { useState } from "react";

interface WelcomeViewProps {
  startButtonText: string;
  onStartCall: (playerName: string) => void;
}

export const WelcomeView = ({
  startButtonText,
  onStartCall,
  ref,
}: React.ComponentProps<"div"> & WelcomeViewProps) => {
  const [playerName, setPlayerName] = useState("");

  return (
    <div ref={ref}>
      <section className="bg-black text-white flex flex-col items-center justify-center h-screen px-4 text-center">
        
        {/* Placeholder for Logo
        <div className="mb-5 w-24 h-24 bg-white/10 rounded-full flex items-center justify-center text-sm text-gray-300">
          LOGO
        </div> */}

        <h1 className="text-3xl font-bold tracking-wide mb-2">
          IMPROV BATTLE
        </h1>

        <p className="text-gray-300 max-w-md text-sm leading-6">
          A voice-powered improv showdown. 
          The host throws unpredictable scenarios. 
          You perform. They react. 
          Survive all rounds to win the spotlight.
        </p>

        <input
          type="text"
          placeholder="Enter your stage name"
          value={playerName}
          onChange={(e) => setPlayerName(e.target.value)}
          className="mt-6 w-64 px-4 py-2 bg-white text-black rounded-lg text-center text-sm outline-none focus:ring-2 focus:ring-white"
        />

        <Button
          variant="primary"
          size="lg"
          className="mt-4 w-64 font-semibold bg-white text-black hover:bg-gray-200"
          onClick={() => onStartCall(playerName)}
        >
          {startButtonText}
        </Button>

        <p className="text-xs text-gray-500 mt-4">
          Voice-only game. 3 short improv rounds. Give it your best performance.
        </p>
      </section>
    </div>
  );
};
