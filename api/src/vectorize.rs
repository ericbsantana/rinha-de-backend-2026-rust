use chrono::{DateTime, Datelike, Timelike, Utc};
use serde::Deserialize;

const MAX_AMOUNT: f64 = 10000.0;
const MAX_INSTALLMENTS: f64 = 12.0;
const AMOUNT_VS_AVG_RATIO: f64 = 10.0;
const MAX_MINUTES: f64 = 1440.0;
const MAX_KM: f64 = 1000.0;
const MAX_TX_COUNT_24H: f64 = 20.0;
const MAX_MERCHANT_AVG_AMOUNT: f64 = 10000.0;

fn mcc_risk(mcc: &str) -> f64 {
    match mcc {
        "5411" => 0.15,
        "5812" => 0.30,
        "5912" => 0.20,
        "5944" => 0.45,
        "7801" => 0.80,
        "7802" => 0.75,
        "7995" => 0.85,
        "4511" => 0.35,
        "5311" => 0.25,
        "5999" => 0.50,
        _ => 0.5, // default pra MCC desconhecido
    }
}

#[derive(Deserialize)]
pub struct Payload {
    id: String,
    transaction: Transaction,
    customer: Customer,
    merchant: Merchant,
    terminal: Terminal,
    last_transaction: Option<LastTransactionDetails>,
}

#[derive(Deserialize)]
pub struct Transaction {
    amount: f64,
    installments: u8,
    requested_at: String,
}

#[derive(Deserialize)]
pub struct Customer {
    avg_amount: f64,
    tx_count_24h: u8,
    known_merchants: Vec<String>,
}

#[derive(Deserialize)]
pub struct Merchant {
    id: String,
    mcc: String,
    avg_amount: f64,
}

#[derive(Deserialize)]
pub struct Terminal {
    is_online: bool,
    card_present: bool,
    km_from_home: f64,
}

#[derive(Deserialize)]
pub struct LastTransactionDetails {
    timestamp: String,
    km_from_current: f64,
}

pub fn vectorize(payload: &Payload) -> [f32; 14] {
    let t = &payload.transaction;
    let c = &payload.customer;
    let m = &payload.merchant;
    let term = &payload.terminal;
    let lt = &payload.last_transaction;
    let mut v = [0.0f32; 14];

    let t_dt: DateTime<Utc> = t.requested_at.parse().unwrap();
    let hour: u32 = t_dt.hour();
    let weekday: u32 = t_dt.weekday().num_days_from_monday();

    v[0] = (t.amount / MAX_AMOUNT).clamp(0.0, 1.0) as f32;
    v[1] = (t.installments as f64 / MAX_INSTALLMENTS).clamp(0.0, 1.0) as f32;
    v[2] = ((t.amount / c.avg_amount) / AMOUNT_VS_AVG_RATIO).clamp(0.0, 1.0) as f32;
    v[3] = (hour as f32 / 23.0).clamp(0.0, 1.0);
    v[4] = (weekday as f32 / 6.0).clamp(0.0, 1.0);
    v[5] = -1.0;
    v[6] = -1.0;

    if let Some(last) = lt {
        let last_dt: DateTime<Utc> = last.timestamp.parse().unwrap();
        let minutes_since = (t_dt - last_dt).num_minutes();
        v[5] = (minutes_since as f64 / MAX_MINUTES).clamp(0.0, 1.0) as f32;
        v[6] = (last.km_from_current / MAX_KM).clamp(0.0, 1.0) as f32;
    }

    v[7] = (term.km_from_home / MAX_KM).clamp(0.0, 1.0) as f32;
    v[8] = (c.tx_count_24h as f64 / MAX_TX_COUNT_24H).clamp(0.0, 1.0) as f32;
    v[9] = if term.is_online { 1.0 } else { 0.0 };
    v[10] = if term.card_present { 1.0 } else { 0.0 };
    v[11] = if c.known_merchants.contains(&m.id) {
        0.0
    } else {
        1.0
    };
    v[12] = mcc_risk(&m.mcc) as f32;
    v[13] = (m.avg_amount / MAX_MERCHANT_AVG_AMOUNT).clamp(0.0, 1.0) as f32;

    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn legitima_da_doc() {
        let payload = Payload {
            id: "tx-1329056812".to_string(),
            transaction: Transaction {
                amount: 41.12,
                installments: 2,
                requested_at: "2026-03-11T18:45:53Z".to_string(),
            },
            customer: Customer {
                avg_amount: 82.24,
                tx_count_24h: 3,
                known_merchants: vec!["MERC-003".to_string(), "MERC-016".to_string()],
            },
            merchant: Merchant {
                id: "MERC-016".to_string(),
                mcc: "5411".to_string(),
                avg_amount: 60.25,
            },
            terminal: Terminal {
                is_online: false,
                card_present: true,
                km_from_home: 29.23,
            },
            last_transaction: None,
        };

        let v = vectorize(&payload);

        let esperado = [
            0.0041, 0.1667, 0.05, 0.7826, // dim 3: 18/23
            0.3333, // dim 4: quarta-feira = 2, 2/6
            -1.0, -1.0, 0.0292, 0.15, 0.0, 1.0, 0.0, 0.15, 0.006,
        ];

        for (i, (atual, esp)) in v.iter().zip(esperado.iter()).enumerate() {
            assert!(
                (atual - esp).abs() < 1e-3,
                "dim {} divergiu: esperado {}, obtido {}",
                i,
                esp,
                atual
            );
        }
    }

    #[test]
    fn with_previous_transaction() {
        let payload = Payload {
            id: "tx-test".to_string(),
            transaction: Transaction {
                amount: 150.0,
                installments: 1,
                requested_at: "2026-03-11T20:00:00Z".to_string(),
            },
            customer: Customer {
                avg_amount: 100.0,
                tx_count_24h: 5,
                known_merchants: vec!["MERC-001".to_string()],
            },
            merchant: Merchant {
                id: "MERC-001".to_string(),
                mcc: "5411".to_string(),
                avg_amount: 200.0,
            },
            terminal: Terminal {
                is_online: true,
                card_present: false,
                km_from_home: 5.0,
            },
            last_transaction: Some(LastTransactionDetails {
                timestamp: "2026-03-11T19:30:00Z".to_string(), // 30 minutos antes
                km_from_current: 250.0,
            }),
        };

        let v = vectorize(&payload);

        // dim 5: 30 minutes / 1440 = 0.02083...
        assert!((v[5] - 0.0208).abs() < 1e-3, "dim 5: got {}", v[5]);

        // dim 6: 250 / 1000 = 0.25
        assert!((v[6] - 0.25).abs() < 1e-3, "dim 6: got {}", v[6]);

        // dim 9: is_online=true -> 1.0
        assert_eq!(v[9], 1.0);

        // dim 10: card_present=false -> 0.0
        assert_eq!(v[10], 0.0);

        // dim 11: merchant in known list -> 0.0 (NOT unknown)
        assert_eq!(v[11], 0.0);
    }
}
