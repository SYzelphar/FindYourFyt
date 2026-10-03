import { useState } from 'react';
import SearchIcon from '@mui/icons-material/Search';
import { DEPARTMENTS } from '../api';

const MODE_TEXT = {
    onboarding: 'Getting to know your style',
    exploring: 'Exploring other styles',
    personalized: 'Personalised for you',
};

const DeckToolbar = ({ department, onDepartment, steers, onSteer, onRemoveSteer, info, disabled }) => {
    const [text, setText] = useState('');

    const submit = (e) => {
        e.preventDefault();
        const value = text.trim();
        if (!value) return;
        onSteer(value);
        setText('');
    };

    return (
        <div className="deckToolbar">
            <div className="deckToolbar__row">
                <div className="segmented" role="group" aria-label="Shopping for">
                    {Object.entries(DEPARTMENTS).map(([key, d]) => (
                        <button key={key} type="button" aria-pressed={department === key}
                            className={department === key ? 'segmented__on' : ''}
                            onClick={() => department !== key && onDepartment(key)}>
                            {d.label}
                        </button>
                    ))}
                </div>
                <form className="steer" onSubmit={submit}>
                    <SearchIcon />
                    <input value={text} onChange={(e) => setText(e.target.value)} maxLength={80}
                        placeholder="Steer it: “in red”, “linen dress”" aria-label="Steer recommendations"
                        disabled={disabled} />
                </form>
            </div>
            <div className="deckToolbar__row deckToolbar__status">
                <span>
                    {MODE_TEXT[info?.mode] || 'Loading…'}
                    {info?.stats?.likes > 0 && ` · ${info.stats.likes} like${info.stats.likes === 1 ? '' : 's'}`}
                    {info?.interests?.length > 0 && ` · your styles: ${info.interests.join(', ')}`}
                </span>
                {steers.map((s) => (
                    <span key={s} className="chip">
                        {s}
                        <button type="button" onClick={() => onRemoveSteer(s)} aria-label={`Remove “${s}”`}>×</button>
                    </span>
                ))}
            </div>
        </div>
    );
};

export default DeckToolbar;
