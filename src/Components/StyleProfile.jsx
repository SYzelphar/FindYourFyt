import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { describeError, fetchProfile, getStoredSessionId, isSessionNotFound } from '../api';

const TRAITS = {
    two_tower_match: ['Follows what similar shoppers buy', 'Goes against the crowd'],
    looks_like_liked: ['Sticks close to favourites', 'Likes variety'],
    popularity: ['Into trending pieces', 'Prefers less-common pieces'],
};

const AFFINITY_LABELS = { productType: 'Types', colour: 'Colours', pattern: 'Patterns', department: 'Departments' };

// "What the recommender has learned about you" for the current swipe session.
const StyleProfile = () => {
    const [profile, setProfile] = useState(null);
    const [error, setError] = useState(null);

    useEffect(() => {
        const sid = getStoredSessionId();
        if (!sid) {
            setError('none');
            return;
        }
        fetchProfile(sid).then(setProfile).catch((err) => setError(isSessionNotFound(err) ? 'none' : describeError(err)));
    }, []);

    let body;
    if (error === 'none' || (profile && profile.swipes === 0)) {
        body = <p>Start <Link to="/">swiping</Link> and your style profile will appear here.</p>;
    } else if (error) {
        body = <p>{error}</p>;
    } else if (!profile) {
        body = <p>Loading…</p>;
    } else {
        const traits = Object.entries(TRAITS)
            .filter(([k]) => Math.abs(profile.learnedWeights[k] ?? 0) > 0)
            .map(([k, [pos, neg]]) => (profile.learnedWeights[k] >= 0.3 ? pos : profile.learnedWeights[k] < 0 ? neg : null))
            .filter(Boolean);
        body = (
            <>
                <div className="style-stats">
                    {[[profile.swipes, 'swipes'], [profile.likes, 'likes'], [profile.passes, 'passes'],
                      [`${Math.round(profile.confidence * 100)}%`, 'model confidence']].map(([v, label]) => (
                        <div key={label} className="style-stat"><strong>{v}</strong><span>{label}</span></div>
                    ))}
                </div>
                {profile.interests.length > 0 && (
                    <div className="style-interests">
                        {profile.interests.map((it) => (
                            <div key={it.label} className="style-interest">
                                <div className="style-interest__thumbs">
                                    {it.items.map((x) => <img key={x.id} src={x.image} alt={x.name} />)}
                                </div>
                                <strong>{it.label}</strong>
                                <span>{Math.round(it.share * 100)}% of your likes</span>
                            </div>
                        ))}
                    </div>
                )}
                <div className="style-affinities">
                    {Object.entries(profile.affinities).map(([key, a]) => (a.loves.length > 0 || a.avoids.length > 0) && (
                        <p key={key}>
                            <strong>{AFFINITY_LABELS[key]}:</strong>{' '}
                            {a.loves.map((x) => x.value).join(', ') || '–'}
                            {a.avoids.length > 0 && <span className="style-avoid"> · not for you: {a.avoids.map((x) => x.value).join(', ')}</span>}
                        </p>
                    ))}
                </div>
                {traits.length > 0 && <p className="style-traits"><strong>Shopping style:</strong> {traits.join(' · ')}</p>}
            </>
        );
    }

    return (
        <div className="profile-section style-profile">
            <h3 className="section-title">Your style profile</h3>
            <div className="section-content">{body}</div>
        </div>
    );
};

export default StyleProfile;
