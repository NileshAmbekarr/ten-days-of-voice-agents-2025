import { Button } from '@/components/livekit/button';
import Image from 'next/image'
import Swiggy from './swiggy-logo.png'

function WelcomeImage() {
  return (
    <div className="mb-4 flex items-center justify-center">
       <Image 
        src={Swiggy} alt="Swiggy Logo" width={300} height={300}
        />
    </div>
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
      <section className="bg-background flex flex-col items-center justify-center text-center px-4">
        <WelcomeImage />

        <h1 className="text-foreground text-2xl font-bold mb-2">
          Swiggy Instamart Voice Ordering
        </h1>

        <p className="text-muted-foreground max-w-prose leading-6 font-medium">
          Grocery shopping made effortless — just speak your order.
          Add items, build carts, or even say things like
          <span className="text-orange-500 font-semibold"> “ingredients for pasta”</span> 
          and I’ll take care of the rest!
        </p>

        <Button
          variant="primary"
          size="lg"
          onClick={onStartCall}
          className="mt-6 w-64 font-semibold bg-orange-500 hover:bg-orange-600 text-white rounded-xl"
        >
          {startButtonText}
        </Button>
      </section>

      <div className="fixed bottom-5 left-0 flex w-full items-center justify-center">
        <p className="text-muted-foreground text-xs font-normal">
          Powered by Voice AI · Swiggy Instamart experience
        </p>
      </div>
    </div>
  );
};
