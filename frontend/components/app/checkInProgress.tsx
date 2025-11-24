export default function CheckinProgress({ data }) {
  return (
    <div className="p-4 w-64 bg-transparent rounded-xl shadow-md border border-[#D4C7B8]">
      <h3 className="font-bold text-lg mb-3 text-[#4B3A2F]">Daily Check-in</h3>

      <div className="space-y-2">
        <div className="flex justify-between">
          <span>Mood</span>
          <span>{data.mood ? "✔" : "..."}</span>
        </div>
        <div className="flex justify-between">
          <span>Energy</span>
          <span>{data.energy ? "✔" : "..."}</span>
        </div>
        <div className="flex justify-between">
          <span>Goals</span>
          <span>{data.goals.length > 0 ? `${data.goals.length} added` : "..."}</span>
        </div>
      </div>
    </div>
  );
}
