import { Button } from '@/components/livekit/button';
import TufIcon from './tuf.png'
import Image from 'next/image';

function TakeUForwardLogo() {
  return (
    <Image
      src={TufIcon}
      width={300}
      height={300}
      alt="Picture of the author"
    />
  );
}

interface WelcomeViewProps {
  startButtonText: string;
  onStartCall: () => void;
}

export const WelcomeView = ({
  startButtonText,
  onStartCall,
  ref,
}: React.ComponentProps<'div'> & WelcomeViewProps) => {
  return (
    <div ref={ref}>
      <section className="bg-background flex flex-col items-center justify-center text-center p-6 rounded-xl">
        
        <TakeUForwardLogo />

        <h1 className="text-2xl font-bold text-primary mb-2">
          TakeUForward – Voice SDR Agent
        </h1>

        <p className="text-foreground max-w-prose pt-1 leading-6 font-medium">
          Learn about TakeUForward, ask your questions, and share your requirements.
        </p>

        <Button
          variant="primary"
          size="lg"
          onClick={onStartCall}
          className="mt-6 w-64 font-semibold bg-blue-500 hover:bg-blue-600"
        >
          {startButtonText}
        </Button>
      </section>

      <div className="fixed bottom-5 left-0 flex w-full items-center justify-center">
        <p className="text-muted-foreground text-xs md:text-sm opacity-70">
          Powered by LiveKit Agents + Murf + Gemini
        </p>
      </div>
    </div>
  );
};
