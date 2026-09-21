import { useEffect, useState } from 'react'

const TIER_LABEL = {
  1: 'Monitor',
  2: 'Revoke session',
  3: 'Hold for approval',
}

const TIER_ACTION = {
  1: 'Runs automatically: alert enriched, session captured, monitoring raised.',
  2: 'Runs automatically and can be undone: session token revoked, re-authentication required.',
  3: 'Staged, waiting on an analyst: account disable and workstation quarantine.',
}

const ASSESSMENT_LABEL = {
  likely_benign: 'Probably fine',
  needs_review: 'Needs a look',
  likely_incident: 'Probably an incident',
  parse_error: 'Explanation unavailable',
}

function DeviationStrip({ series, flaggedDate, tier }) {
  if (!series || series.length === 0) return null

  const counts = series.map((d) => d.n)
  const max = Math.max(...counts)
  const sorted = [...counts].sort((a, b) => a - b)
  const median = sorted[Math.floor(sorted.length / 2)]

  const W = 220
  const H = 42
  const gap = 1
  const barW = Math.max(1.5, W / series.length - gap)

  return (
    <svg
      className="strip"
      viewBox={`0 0 ${W} ${H}`}
      role="img"
      aria-label={`Daily access counts. Median ${median}, flagged day ${
        series.find((d) => d.date === flaggedDate)?.n ?? 0
      }.`}
    >
      <line
        x1="0"
        x2={W}
        y1={H - (median / max) * (H - 4) - 2}
        y2={H - (median / max) * (H - 4) - 2}
        className="strip-median"
      />
      {series.map((d, i) => {
        const h = Math.max(1, (d.n / max) * (H - 4))
        const isFlagged = d.date === flaggedDate
        return (
          <rect
            key={d.date}
            x={i * (barW + gap)}
            y={H - h - 2}
            width={barW}
            height={h}
            className={isFlagged ? `strip-bar flagged tier-${tier}` : 'strip-bar'}
          />
        )
      })}
    </svg>
  )
}

function Measure({ label, value, note }) {
  return (
    <div className="measure">
      <dt>{label}</dt>
      <dd>
        <span className="measure-value">{value}</span>
        {note && <span className="measure-note">{note}</span>}
      </dd>
    </div>
  )
}

function pct(v) {
  return `${Math.round(v * 100)}%`
}

function Detail({ alert }) {
  if (!alert) {
    return (
      <section className="detail detail-empty">
        <p>Select an alert to see why it was raised.</p>
      </section>
    )
  }

  const m = alert.measures
  const d = alert.deviation
  const e = alert.explanation
  const median =
    alert.series.length > 0
      ? [...alert.series.map((s) => s.n)].sort((a, b) => a - b)[
          Math.floor(alert.series.length / 2)
        ]
      : 0

  return (
    <section className="detail">
      <header className="detail-head">
        <div>
          <h2>{alert.user_id}</h2>
          <p className="detail-sub">
            {alert.role.replace('_', ' ')} in {alert.department} · {alert.date}
          </p>
        </div>
        <div className={`tier-badge tier-${alert.tier}`}>
          <span className="tier-num">Tier {alert.tier}</span>
          <span className="tier-name">{TIER_LABEL[alert.tier]}</span>
        </div>
      </header>

      <p className="tier-action">{TIER_ACTION[alert.tier]}</p>
      <ul className="tier-reasons">
        {alert.tier_reasons.map((r) => (
          <li key={r}>{r}</li>
        ))}
      </ul>

      <div className="block">
        <h3>That day, against their own thirty days</h3>
        <DeviationStrip
          series={alert.series}
          flaggedDate={alert.date}
          tier={alert.tier}
        />
        <p className="strip-caption">
          {m.n_events} accesses on {alert.date}. This person's median day is{' '}
          {median}.
        </p>
      </div>

      <div className="block">
        <h3>What the day looked like</h3>
        <dl className="measures">
          <Measure
            label="Records opened"
            value={m.n_events}
            note={`${d.events > 0 ? '+' : ''}${d.events} vs own normal`}
          />
          <Measure label="Distinct patients" value={m.distinct_patients} />
          <Measure label="Units touched" value={m.units} />
          <Measure
            label="Outside own unit"
            value={pct(m.cross_dept_ratio)}
            note={`${d.cross_dept > 0 ? '+' : ''}${d.cross_dept} vs own normal`}
          />
          <Measure label="Units rarely visited" value={pct(m.rare_dept_ratio)} />
          <Measure label="Outside usual hours" value={pct(m.off_hours_ratio)} />
          <Measure
            label="Unfamiliar workstation"
            value={pct(m.new_workstation_ratio)}
          />
          <Measure label="Exports" value={pct(m.export_ratio)} />
          <Measure label="Busiest single hour" value={m.peak_hour_events} />
        </dl>
      </div>

      {e ? (
        <div className="block">
          <h3>Analyst summary</h3>
          <p className={`assessment assessment-${e.assessment}`}>
            {ASSESSMENT_LABEL[e.assessment] ?? e.assessment}
            <span className="confidence">{e.confidence} confidence</span>
          </p>
          <p className="summary">{e.summary}</p>
          {e.reasoning && <p className="reasoning">{e.reasoning}</p>}
          {e.hipaa_concern && (
            <p className="hipaa">HIPAA {e.hipaa_concern}</p>
          )}
          {e.recommended_actions?.length > 0 && (
            <>
              <h4>What to do next</h4>
              <ul className="actions">
                {e.recommended_actions.map((a) => (
                  <li key={a}>{a}</li>
                ))}
              </ul>
            </>
          )}
        </div>
      ) : (
        <div className="block">
          <h3>Analyst summary</h3>
          <p className="no-explanation">
            Not generated yet. Run <code>explain.py</code> to add one.
          </p>
        </div>
      )}

      <footer className="ground-truth">
        {alert.is_anomaly
          ? `Evaluation label: ${alert.attack_type.replace(/_/g, ' ')}`
          : 'Evaluation label: no attack on this day'}
      </footer>
    </section>
  )
}

export default function App() {
  const [data, setData] = useState(null)
  const [error, setError] = useState(null)
  const [selected, setSelected] = useState(null)
  const [showTruth, setShowTruth] = useState(false)

  useEffect(() => {
    fetch('data.json')
      .then((r) => {
        if (!r.ok) throw new Error(`data.json returned ${r.status}`)
        return r.json()
      })
      .then((d) => {
        setData(d)
        setSelected(d.alerts[0]?.id ?? null)
      })
      .catch((e) => setError(e.message))
  }, [])

  if (error) {
    return (
      <main className="state">
        <h1>No data to show</h1>
        <p>
          {error}. Generate it with{' '}
          <code>python src/dashboard/export_data.py</code>, then reload.
        </p>
      </main>
    )
  }

  if (!data) {
    return (
      <main className="state">
        <p>Loading alerts…</p>
      </main>
    )
  }

  const { metrics, generated_from: src, alerts } = data
  const current = alerts.find((a) => a.id === selected)

  return (
    <div className="app">
      <header className="masthead">
        <div className="masthead-title">
          <h1>Access review</h1>
          <p>
            {alerts.length} days flagged out of {src.user_days.toLocaleString()}{' '}
            reviewed, across {src.users} staff over {src.days} days.
          </p>
        </div>
        <dl className="scoreboard">
          <div>
            <dt>Accounts caught</dt>
            <dd>
              {metrics.accounts_caught} of {metrics.accounts_total}
            </dd>
          </div>
          <div>
            <dt>Alerts that were real</dt>
            <dd>{pct(metrics.precision)}</dd>
          </div>
          <div>
            <dt>Attack days found</dt>
            <dd>{pct(metrics.recall)}</dd>
          </div>
        </dl>
      </header>

      <div className="panes">
        <section className="list">
          <div className="list-head">
            <h2>Flagged days</h2>
            <label className="truth-toggle">
              <input
                type="checkbox"
                checked={showTruth}
                onChange={(e) => setShowTruth(e.target.checked)}
              />
              Show evaluation labels
            </label>
          </div>

          <ol>
            {alerts.map((a) => (
              <li key={a.id}>
                <button
                  className={a.id === selected ? 'row selected' : 'row'}
                  onClick={() => setSelected(a.id)}
                  aria-current={a.id === selected}
                >
                  <span className={`pip tier-${a.tier}`} aria-hidden="true" />
                  <span className="row-who">
                    <strong>{a.user_id}</strong>
                    <span className="row-role">
                      {a.role.replace('_', ' ')}, {a.department}
                    </span>
                  </span>
                  <span className="row-date">{a.date}</span>
                  <DeviationStrip
                    series={a.series}
                    flaggedDate={a.date}
                    tier={a.tier}
                  />
                  {showTruth && (
                    <span
                      className={
                        a.is_anomaly ? 'truth truth-hit' : 'truth truth-miss'
                      }
                    >
                      {a.is_anomaly
                        ? a.attack_type.replace(/_/g, ' ')
                        : 'no attack'}
                    </span>
                  )}
                </button>
              </li>
            ))}
          </ol>
        </section>

        <Detail alert={current} />
      </div>
    </div>
  )
}
