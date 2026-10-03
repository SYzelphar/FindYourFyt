import React, { useCallback, useEffect, useRef, useState } from 'react';
import TinderCard from 'react-tinder-card';
import { useSearchParams } from 'react-router-dom';
import '../Swipe.css';
import SwipeButtons from './SwipeButtons';
import DeckToolbar from './DeckToolbar';
import WhyPanel from './WhyPanel';
import NoDrag from './NoDrag';
import { useCart } from '../useCart';
import {
    addSteer,
    createSession,
    describeError,
    fetchRecommendations,
    getStoredDepartment,
    getStoredSessionId,
    isSessionNotFound,
    removeSteer,
    sendSwipe,
    storeDepartment,
    storeSessionId,
} from '../api';

// The card queue is owned by the recommendation engine: every swipe is sent to
// the backend and the response replaces the queue, so the next card is always
// the top recommendation for the updated preference model.
const Swipe = () => {
    const [queue, setQueue] = useState([]);      // upcoming recommendations, queue[0] is on screen
    const [leaving, setLeaving] = useState([]);  // swiped cards still animating off screen
    const [pending, setPending] = useState(false);
    const [status, setStatus] = useState('loading'); // loading | ready | error
    const [notice, setNotice] = useState(null);
    const [info, setInfo] = useState(null);      // mode, interests, stats from the last response
    const [steers, setSteers] = useState([]);
    const [department, setDepartment] = useState(getStoredDepartment);
    const [whyId, setWhyId] = useState(null);
    const [searchParams, setSearchParams] = useSearchParams();

    const sessionIdRef = useRef(null);
    const seenRef = useRef(new Set());     // ids swiped in this tab, never shown again
    const brokenImagesRef = useRef(new Set());
    const cardRef = useRef(null);
    const { addToCart } = useCart();

    const acceptRecommendations = useCallback((items) =>
        (items || []).filter((p) => !seenRef.current.has(p.id) && !brokenImagesRef.current.has(p.id)),
    []);

    const applyDeck = useCallback((data) => {
        setQueue(acceptRecommendations(data.recommendations));
        setInfo({ mode: data.mode, interests: data.interests, stats: data.stats });
        setSteers(data.steers || []);
        setWhyId(null);
    }, [acceptRecommendations]);

    const adoptSession = (id) => {
        sessionIdRef.current = id;
        storeSessionId(id);
    };

    // Load: resume this tab's session if the server still has it, else start one.
    useEffect(() => {
        let cancelled = false;
        (async () => {
            try {
                let data;
                const stored = getStoredSessionId();
                if (stored) {
                    try {
                        data = await fetchRecommendations(stored);
                    } catch (err) {
                        if (!isSessionNotFound(err)) throw err;
                        data = await createSession(department);
                    }
                } else {
                    data = await createSession(department);
                }
                if (cancelled) return;
                adoptSession(data.session_id);
                applyDeck(data);
                setStatus('ready');
            } catch (err) {
                if (cancelled) return;
                console.error('Failed to load recommendations', err);
                setNotice(describeError(err));
                setStatus('error');
            }
        })();
        return () => { cancelled = true; };
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [applyDeck]);

    // Preload images of queued cards; drop any product whose image fails to load.
    useEffect(() => {
        queue.forEach((product) => {
            const img = new Image();
            img.onerror = () => {
                brokenImagesRef.current.add(product.id);
                setQueue((q) => q.filter((p) => p.id !== product.id));
            };
            img.src = product.image;
        });
    }, [queue]);

    // Run a session request; if the server forgot the session (restart), start a new one and retry.
    const withSession = async (request) => {
        try {
            return await request(sessionIdRef.current);
        } catch (err) {
            if (!isSessionNotFound(err)) throw err;
            adoptSession((await createSession(department)).session_id);
            return request(sessionIdRef.current);
        }
    };

    // Single code path for drag swipes and button presses.
    const handleSwipe = async (direction, product) => {
        if (direction !== 'left' && direction !== 'right') return;
        seenRef.current.add(product.id);
        setLeaving((l) => [...l, product]);
        const fallback = queue.filter((p) => p.id !== product.id);
        setQueue(fallback);
        setPending(true);
        try {
            const data = await withSession((sid) => sendSwipe(sid, product.id, direction));
            if (acceptRecommendations(data.recommendations).length) {
                applyDeck(data);
                setNotice(null);
            } else {
                applyDeck(await fetchRecommendations(sessionIdRef.current));
            }
        } catch (err) {
            console.error('Swipe feedback failed', err);
            setNotice(`${describeError(err)} Showing your earlier recommendations.`);
            setQueue(acceptRecommendations(fallback));
        } finally {
            setPending(false);
        }
    };

    const runDeckRequest = async (request) => {
        if (pending) return;
        setPending(true);
        try {
            applyDeck(await withSession(request));
            setNotice(null);
        } catch (err) {
            setNotice(describeError(err));
        } finally {
            setPending(false);
        }
    };

    const changeDepartment = async (key) => {
        setDepartment(key);
        storeDepartment(key);
        seenRef.current = new Set();
        await runDeckRequest(async () => {
            const data = await createSession(key);
            adoptSession(data.session_id);
            return data;
        });
    };

    // Category links on the Home page open the deck as /?steer=<category>.
    const steerParam = searchParams.get('steer');
    useEffect(() => {
        if (status !== 'ready' || !steerParam) return;
        setSearchParams({}, { replace: true });
        runDeckRequest((sid) => addSteer(sid, steerParam));
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [status, steerParam]);

    const onCardLeftScreen = (id) => setLeaving((l) => l.filter((p) => p.id !== id));

    const swipeWithButton = async (direction) => {
        if (pending || !queue.length || !cardRef.current) return;
        await cardRef.current.swipe(direction); // fires onSwipe -> handleSwipe
    };

    const current = pending ? null : queue[0];
    const cards = [...(current ? [current] : []), ...leaving];

    return (
        <>
            <DeckToolbar
                department={department}
                onDepartment={changeDepartment}
                steers={steers}
                onSteer={(text) => runDeckRequest((sid) => addSteer(sid, text))}
                onRemoveSteer={(text) => runDeckRequest((sid) => removeSteer(sid, text))}
                info={info}
                disabled={status !== 'ready'}
            />

            <div className="cardContainer">
                {cards.map((product) => {
                    const isActive = current && product.id === current.id;
                    const headline = product.explanation?.headline;
                    return (
                        <TinderCard
                            ref={isActive ? cardRef : null}
                            className="swipe"
                            key={product.id}
                            preventSwipe={['up', 'down']}
                            onSwipe={(dir) => isActive && handleSwipe(dir, product)}
                            onCardLeftScreen={() => onCardLeftScreen(product.id)}
                        >
                            <div
                                style={{ backgroundImage: `url(${product.image})` }}
                                className="card"
                                data-product-id={product.id}
                            >
                                <div className="card__top">
                                    {headline && <span className="card__reason">{headline}</span>}
                                    {isActive && (
                                        <NoDrag as="span">
                                            <button className="card__why" aria-expanded={whyId === product.id}
                                                onClick={() => setWhyId(whyId === product.id ? null : product.id)}>
                                                Why?
                                            </button>
                                        </NoDrag>
                                    )}
                                </div>
                                {isActive && whyId === product.id && (
                                    <WhyPanel product={product} onClose={() => setWhyId(null)} />
                                )}
                                <div className="card__caption">
                                    <h3>{product.name}</h3>
                                    <p className="card__sub">{product.colour} · {product.productType}</p>
                                </div>
                            </div>
                        </TinderCard>
                    );
                })}

                {!current && !leaving.length && (
                    <div className="card card--placeholder">
                        {status === 'loading' || pending ? (
                            <p>Finding your fit…</p>
                        ) : status === 'error' ? (
                            <>
                                <p>{notice}</p>
                                <button className="card__retry" onClick={() => window.location.reload()}>Try again</button>
                            </>
                        ) : (
                            <p>You&apos;ve seen everything we have for now.</p>
                        )}
                    </div>
                )}
            </div>

            {notice && status === 'ready' && <div className="swipeNotice">{notice}</div>}

            <SwipeButtons
                disabled={pending || !current}
                onDislike={() => swipeWithButton('left')}
                onLike={() => swipeWithButton('right')}
                onCart={() => current && addToCart(current)}
            />
        </>
    );
};

export default Swipe;
