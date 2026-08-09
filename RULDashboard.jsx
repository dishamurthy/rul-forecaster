import { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  Legend, ReferenceLine, Area, AreaChart, ResponsiveContainer
} from "recharts";

// ── Design: dark aerospace telemetry aesthetic
// Palette: near-black bg, amber primary (instrument panel), cyan accent (live data)
// Typography: monospace data readouts, sans-serif labels
// Signature: degradation curve with shaded MC uncertainty band

const C = {
  bg:       "#0a0c0f",
  panel:    "#111418",
  border:   "#1e2530",
  amber:    "#f5a623",
  cyan:     "#00d4ff",
  green:    "#22c55e",
  red:      "#ef4444",
  muted:    "#4a5568",
  text:     "#e2e8f0",
  subtext:  "#718096",
};

const mono = "'JetBrains Mono', 'Fira Code', 'Courier New', monospace";
const sans = "'Inter', system-ui, sans-serif";

// ── Stat tile ──────────────────────────────────────────────────────────────────
function StatTile({ label, value, unit = "", color = C.amber, trend = null }) {
  return (
    <div style={{
      background: C.panel,
      border: `1px solid ${C.border}`,
      borderRadius: 8,
      padding: "14px 18px",
      minWidth: 130,
    }}>
      <div style={{ fontFamily: sans, fontSize: 10, color: C.subtext,
                    letterSpacing: "0.12em", textTransform: "uppercase", marginBottom: 6 }}>
        {label}
      </div>
      <div style={{ fontFamily: mono, fontSize: 26, fontWeight: 700, color,
                    letterSpacing: "-0.02em", lineHeight: 1 }}>
        {value !== null && value !== undefined ? value : "—"}
        <span style={{ fontSize: 13, fontWeight: 400, marginLeft: 3, color: C.subtext }}>
          {unit}
        </span>
      </div>
      {trend !== null && (
        <div style={{ fontFamily: mono, fontSize: 11, marginTop: 4,
                      color: trend > 0 ? C.red : C.green }}>
          {trend > 0 ? "▲" : "▼"} {Math.abs(trend).toFixed(1)}
        </div>
      )}
    </div>
  );
}

// ── Sensor bar ─────────────────────────────────────────────────────────────────
function SensorBar({ label, value, max, color = C.cyan }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0;
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between",
                    fontFamily: mono, fontSize: 11, color: C.subtext, marginBottom: 3 }}>
        <span>{label}</span>
        <span style={{ color: C.text }}>{value?.toFixed(1) ?? "—"}</span>
      </div>
      <div style={{ height: 4, background: C.border, borderRadius: 2 }}>
        <div style={{
          height: "100%", width: `${pct}%`,
          background: color, borderRadius: 2,
          transition: "width 0.4s ease",
        }} />
      </div>
    </div>
  );
}

// ── Custom tooltip ─────────────────────────────────────────────────────────────
function RULTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null;
  return (
    <div style={{
      background: C.panel, border: `1px solid ${C.border}`,
      borderRadius: 6, padding: "10px 14px", fontFamily: mono, fontSize: 12,
    }}>
      <div style={{ color: C.subtext, marginBottom: 6 }}>Lap {label}</div>
      {payload.map((p, i) => (
        <div key={i} style={{ color: p.color, marginBottom: 2 }}>
          {p.name}: {typeof p.value === "number" ? p.value.toFixed(1) : p.value}
        </div>
      ))}
    </div>
  );
}

// ── Main dashboard ─────────────────────────────────────────────────────────────
export default function RULDashboard() {
  const [data,       setData]       = useState([]);
  const [latest,     setLatest]     = useState(null);
  const [status,     setStatus]     = useState("IDLE");
  const [source,     setSource]     = useState("synthetic");
  const [connected,  setConnected]  = useState(false);
  const [alertMsg,   setAlertMsg]   = useState(null);
  const wsRef = useRef(null);

  const connect = useCallback((src) => {
    if (wsRef.current) wsRef.current.close();
    setData([]);
    setLatest(null);
    setAlertMsg(null);
    setStatus("CONNECTING");

    // In real deployment: ws://localhost:8000/ws/telemetry/${src}
    // For demo, simulate the WebSocket with synthetic data
    setStatus("LIVE");
    setConnected(true);
    runSimulation(src);
  }, []);

  // Simulation for standalone demo (no backend required)
  const simRef = useRef(null);
  const runSimulation = (src) => {
    if (simRef.current) clearInterval(simRef.current);
    let lap = 1;
    const maxLaps = 40;

    simRef.current = setInterval(() => {
      if (lap > maxLaps) {
        clearInterval(simRef.current);
        setStatus("COMPLETE");
        setConnected(false);
        return;
      }

      const tyreAge = lap - 1;
      const degradation = lap > 22 ? (lap - 22) * 0.09 : 0;
      const noise = () => (Math.random() - 0.5) * 0.04;

      // Simulate sensor degradation curve
      const speed     = 298 - degradation * 2.5  + (Math.random()-0.5)*3;
      const rpm       = 11400 - degradation * 60 + (Math.random()-0.5)*150;
      const throttle  = 0.83 - degradation * 0.005 + noise();
      const brake     = 0.14 + degradation * 0.004 + noise();
      const lapDelta  = degradation + (Math.random()-0.5)*0.12;

      // Simulate MC Dropout uncertainty (grows as degradation increases)
      const trueRUL   = Math.max(0, maxLaps - lap);
      const uncertainty = 1.5 + degradation * 2.5;
      const rulMean   = trueRUL + (Math.random()-0.5)*uncertainty;
      const ciLo      = Math.max(0, rulMean - uncertainty * 1.645);
      const ciHi      = rulMean + uncertainty * 1.645;

      const point = {
        lap, tyreAge,
        speed: +speed.toFixed(1),
        rpm:   +rpm.toFixed(0),
        throttle: +throttle.toFixed(3),
        brake: +brake.toFixed(3),
        lapDelta: +lapDelta.toFixed(3),
        trueRUL,
        rulMean: +rulMean.toFixed(1),
        ciLo:    +ciLo.toFixed(1),
        ciHi:    +ciHi.toFixed(1),
        hi: +(Math.max(0, 1 - degradation * 0.08) * (0.95 + noise())).toFixed(3),
      };

      setLatest(point);
      setData(prev => [...prev.slice(-50), point]);

      // Alert: RUL < 8 laps
      if (trueRUL <= 8 && trueRUL > 0) {
        setAlertMsg(`⚠ RUL CRITICAL: ${trueRUL} laps remaining`);
      } else {
        setAlertMsg(null);
      }

      lap++;
    }, 900);
  };

  useEffect(() => () => {
    if (simRef.current) clearInterval(simRef.current);
    if (wsRef.current)  wsRef.current.close();
  }, []);

  const hiColor = latest?.hi > 0.7 ? C.green : latest?.hi > 0.4 ? C.amber : C.red;
  const rulColor = latest?.trueRUL > 15 ? C.green : latest?.trueRUL > 8 ? C.amber : C.red;

  // Build chart data: area bands for CI
  const chartData = data.map(d => ({
    lap:     d.lap,
    "True RUL": d.trueRUL,
    "RUL (predicted)": d.rulMean,
    ciLo:    d.ciLo,
    ciHi:    d.ciHi,
    hi:      d.hi,
  }));

  return (
    <div style={{
      background: C.bg, minHeight: "100vh", color: C.text,
      fontFamily: sans, padding: "20px 24px",
    }}>
      {/* Header */}
      <div style={{ display: "flex", alignItems: "center",
                    justifyContent: "space-between", marginBottom: 20 }}>
        <div>
          <div style={{ fontFamily: mono, fontSize: 11, color: C.amber,
                        letterSpacing: "0.18em", marginBottom: 4 }}>
            PREDICTIVE HEALTH MANAGEMENT
          </div>
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700, letterSpacing: "-0.02em" }}>
            RUL Forecaster
            <span style={{ fontSize: 13, fontWeight: 400, color: C.subtext, marginLeft: 10 }}>
              LSTM + MC Dropout · {source === "f1" ? "FastF1 Telemetry" : "Synthetic Stint"}
            </span>
          </h1>
        </div>

        <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
          <div style={{
            width: 8, height: 8, borderRadius: "50%",
            background: connected ? C.green : C.muted,
            boxShadow: connected ? `0 0 8px ${C.green}` : "none",
          }} />
          <span style={{ fontFamily: mono, fontSize: 11,
                         color: connected ? C.green : C.subtext }}>
            {status}
          </span>
          <button
            onClick={() => connect(source)}
            style={{
              background: connected ? "transparent" : C.amber,
              color: connected ? C.subtext : "#000",
              border: connected ? `1px solid ${C.border}` : "none",
              borderRadius: 6, padding: "7px 16px", cursor: "pointer",
              fontFamily: mono, fontSize: 12, fontWeight: 600,
            }}
          >
            {connected ? "Restart" : "▶ Start Stream"}
          </button>

          <select
            value={source}
            onChange={e => setSource(e.target.value)}
            style={{
              background: C.panel, color: C.text, border: `1px solid ${C.border}`,
              borderRadius: 6, padding: "7px 10px", fontFamily: mono, fontSize: 12,
            }}
          >
            <option value="synthetic">Synthetic Stint</option>
            <option value="f1">FastF1 (2023 Bahrain)</option>
          </select>
        </div>
      </div>

      {/* Alert banner */}
      {alertMsg && (
        <div style={{
          background: "#2d0a0a", border: `1px solid ${C.red}`,
          borderRadius: 8, padding: "10px 16px", marginBottom: 16,
          fontFamily: mono, fontSize: 13, color: C.red,
          display: "flex", alignItems: "center", gap: 8,
        }}>
          {alertMsg} — Consider pit stop
        </div>
      )}

      {/* KPI row */}
      <div style={{ display: "flex", gap: 12, marginBottom: 20, flexWrap: "wrap" }}>
        <StatTile label="Predicted RUL" value={latest?.rulMean?.toFixed(1) ?? "—"} unit="laps" color={rulColor} />
        <StatTile label="True RUL"      value={latest?.trueRUL ?? "—"}               unit="laps" color={C.subtext} />
        <StatTile label="Health Index"  value={latest?.hi?.toFixed(3) ?? "—"}        color={hiColor} />
        <StatTile label="Tyre Age"      value={latest?.tyreAge ?? "—"}               unit="laps" color={C.cyan} />
        <StatTile label="Lap Δ Time"    value={latest?.lapDelta?.toFixed(3) ?? "—"}  unit="s"
                  color={latest?.lapDelta > 1.5 ? C.red : C.green} />
        <StatTile label="90% CI Width"
                  value={latest ? (latest.ciHi - latest.ciLo).toFixed(1) : "—"}
                  unit="laps" color={C.muted} />
      </div>

      {/* Main chart: RUL curve + CI band */}
      <div style={{
        background: C.panel, border: `1px solid ${C.border}`,
        borderRadius: 10, padding: "18px 16px 10px", marginBottom: 16,
      }}>
        <div style={{ fontFamily: sans, fontSize: 12, fontWeight: 600,
                      color: C.subtext, marginBottom: 12, letterSpacing: "0.04em" }}>
          REMAINING USEFUL LIFE — MC Dropout 90% Confidence Interval
        </div>

        {data.length === 0 ? (
          <div style={{ height: 220, display: "flex", alignItems: "center",
                        justifyContent: "center", color: C.muted, fontFamily: mono, fontSize: 13 }}>
            Press ▶ Start Stream to begin telemetry ingestion
          </div>
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <AreaChart data={chartData} margin={{ top: 5, right: 10, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="ciGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%"  stopColor={C.amber} stopOpacity={0.15} />
                  <stop offset="95%" stopColor={C.amber} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid stroke={C.border} strokeDasharray="3 3" />
              <XAxis dataKey="lap" tick={{ fontFamily: mono, fontSize: 10, fill: C.subtext }}
                     label={{ value: "Lap", position: "insideBottom", offset: -2,
                              fill: C.subtext, fontSize: 10 }} />
              <YAxis tick={{ fontFamily: mono, fontSize: 10, fill: C.subtext }}
                     label={{ value: "RUL (laps)", angle: -90, position: "insideLeft",
                              fill: C.subtext, fontSize: 10 }} />
              <Tooltip content={<RULTooltip />} />
              <ReferenceLine y={8} stroke={C.red} strokeDasharray="4 4"
                             label={{ value: "Critical", fill: C.red, fontSize: 10, fontFamily: mono }} />

              {/* CI band — upper */}
              <Area dataKey="ciHi" stroke="none" fill="url(#ciGrad)"
                    name="CI Upper 90%" legendType="none" />
              {/* CI band — lower */}
              <Area dataKey="ciLo" stroke="none" fill={C.bg}
                    name="CI Lower 90%" legendType="none" />

              <Line dataKey="True RUL"        stroke={C.subtext} strokeWidth={1.5}
                    dot={false} strokeDasharray="4 3" />
              <Line dataKey="RUL (predicted)" stroke={C.amber}   strokeWidth={2.5}
                    dot={false} activeDot={{ r: 4 }} />
              <Legend wrapperStyle={{ fontFamily: mono, fontSize: 11, paddingTop: 8 }} />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>

      {/* Bottom row: sensor bars + HI trend */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 14 }}>

        {/* Sensor readouts */}
        <div style={{
          background: C.panel, border: `1px solid ${C.border}`,
          borderRadius: 10, padding: "16px 18px",
        }}>
          <div style={{ fontFamily: sans, fontSize: 12, fontWeight: 600,
                        color: C.subtext, marginBottom: 14, letterSpacing: "0.04em" }}>
            SENSOR CHANNELS
          </div>
          <SensorBar label="Speed (km/h)"  value={latest?.speed}    max={340}   color={C.cyan}  />
          <SensorBar label="RPM"           value={latest?.rpm}      max={12500} color={C.amber} />
          <SensorBar label="Throttle"      value={latest?.throttle} max={1}     color={C.green} />
          <SensorBar label="Brake"         value={latest?.brake}    max={1}
                     color={latest?.brake > 0.3 ? C.red : C.cyan} />
        </div>

        {/* Health Index trend */}
        <div style={{
          background: C.panel, border: `1px solid ${C.border}`,
          borderRadius: 10, padding: "16px 18px",
        }}>
          <div style={{ fontFamily: sans, fontSize: 12, fontWeight: 600,
                        color: C.subtext, marginBottom: 14, letterSpacing: "0.04em" }}>
            COMPOSITE HEALTH INDEX
          </div>
          {data.length === 0 ? (
            <div style={{ height: 120, display: "flex", alignItems: "center",
                          justifyContent: "center", color: C.muted, fontFamily: mono, fontSize: 12 }}>
              Awaiting data...
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={130}>
              <AreaChart data={chartData.slice(-25)} margin={{ top: 5, right: 5, bottom: 0, left: -10 }}>
                <defs>
                  <linearGradient id="hiGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%"  stopColor={C.green} stopOpacity={0.3} />
                    <stop offset="95%" stopColor={C.green} stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <XAxis dataKey="lap" tick={{ fontFamily: mono, fontSize: 9, fill: C.subtext }} />
                <YAxis domain={[0, 1]} tick={{ fontFamily: mono, fontSize: 9, fill: C.subtext }} />
                <Tooltip content={<RULTooltip />} />
                <ReferenceLine y={0.5} stroke={C.amber} strokeDasharray="3 3" />
                <Area dataKey="hi" stroke={C.green} strokeWidth={2}
                      fill="url(#hiGrad)" name="Health Index" dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>

      {/* Footer */}
      <div style={{ marginTop: 18, fontFamily: mono, fontSize: 10, color: C.muted,
                    display: "flex", justifyContent: "space-between" }}>
        <span>LSTM · PyTorch · MC Dropout UQ · CMAPSS Benchmark · FastF1</span>
        <span>Lap {latest?.lap ?? 0} / Tyre age {latest?.tyreAge ?? 0}</span>
      </div>
    </div>
  );
}
