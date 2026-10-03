import NoDrag from './NoDrag';

// Plain-language names for the recommender's score components (backend/recsys/session_model.py).
const PARTS = {
    style_learned_from_swipes: 'Your swipe history',
    two_tower_match: 'Shoppers with your taste',
    looks_like_liked: 'Looks like your likes',
    matches_interest: 'Fits one of your styles',
    looks_like_passed: 'Looks like your passes',
    popularity: 'Trending',
    steer: 'Your search words',
};

const SOURCES = {
    taste_profile: 'your taste profile',
    learned_style: 'your learned style',
    steer: 'your search words',
    explore: 'exploration',
    trending: 'trending items',
};

const sourceLabel = (s, interest) => (s.startsWith('interest:') ? `interest “${interest || 'one of your styles'}”` : SOURCES[s] || s);

const WhyPanel = ({ product, onClose }) => {
    const e = product.explanation || {};
    const parts = Object.entries(e.contributions || {})
        .filter(([k, v]) => PARTS[k] && Math.abs(v) >= 0.05)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    const scale = Math.max(0.5, ...parts.map(([, v]) => Math.abs(v)));
    const pct = Math.min(99, Math.round((product.pLike ?? 0) * 100));   // never claim certainty

    return (
        <NoDrag className="why">
            <button className="why__close" onClick={onClose} aria-label="Close explanation">×</button>
            <p className="why__headline">{e.headline}</p>

            <div className="why__row">
                <span className="why__label">Predicted match</span>
                <div className="why__meter" role="meter" aria-valuenow={pct} aria-valuemin={0} aria-valuemax={100}>
                    <div className="why__meterFill" style={{ width: `${pct}%` }} />
                </div>
                <span className="why__value">{pct}%</span>
            </div>

            {e.becauseYouLiked?.length > 0 && (
                <div className="why__section">
                    <span className="why__label">Because you liked</span>
                    <div className="why__likes">
                        {e.becauseYouLiked.map((x) => (
                            <figure key={x.id} className="why__like">
                                <img src={x.image} alt={x.name} />
                                <figcaption>{x.name} · {Math.round(x.similarity * 100)}% similar</figcaption>
                            </figure>
                        ))}
                    </div>
                </div>
            )}

            {parts.length > 0 && (
                <div className="why__section">
                    <span className="why__label">What drove the score</span>
                    <ul className="why__bars">
                        {parts.map(([k, v]) => (
                            <li key={k} title={`${PARTS[k]}: ${v > 0 ? '+' : ''}${v.toFixed(2)} (log-odds)`}>
                                <span className="why__barName">{PARTS[k]}</span>
                                <span className="why__barTrack">
                                    <span className={`why__bar ${v >= 0 ? 'why__bar--pos' : 'why__bar--neg'}`}
                                        style={{ width: `${(Math.abs(v) / scale) * 100}%` }} />
                                </span>
                                <span className="why__barValue">{v > 0 ? '+' : '−'}{Math.abs(v).toFixed(1)}</span>
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            <p className="why__meta">
                Found via {(product.sources || []).map((s) => sourceLabel(s, e.interest)).join(', ')}
                {e.sharedAttributes?.length > 0 && <> · shares {e.sharedAttributes.join(', ').toLowerCase()} with a like</>}
            </p>
        </NoDrag>
    );
};

export default WhyPanel;
