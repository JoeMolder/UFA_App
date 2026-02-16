#!/usr/bin/env python3
"""
Check if turnover coordinate averaging is creating a midfield clustering problem.
"""

import psycopg2
from psycopg2.extras import RealDictCursor

DB_CONFIG = {
    'dbname': 'ufa_analytics',
    'user': 'joemolder',
    'password': '',
    'host': 'localhost',
    'port': 5432
}

def check_averaging_issue():
    conn = psycopg2.connect(**DB_CONFIG, cursor_factory=RealDictCursor)
    cur = conn.cursor()

    print("=" * 70)
    print("CHECKING FOR MIDFIELD CLUSTERING ISSUE")
    print("=" * 70)

    # Get turnovers and their subsequent throws
    cur.execute("""
        WITH turnovers AS (
            SELECT
                e1.game_id,
                e1.event_number,
                e1.event_type,
                e1.team as turnover_team,
                CASE
                    WHEN e1.event_type = 20 THEN e1.receiver_y
                    WHEN e1.event_type = 22 THEN e1.turnover_y
                END as turnover_y,
                -- Get the next event from opposing team
                (
                    SELECT e2.thrower_y
                    FROM events e2
                    WHERE e2.game_id = e1.game_id
                        AND e2.event_number > e1.event_number
                        AND e2.team != e1.team
                        AND e2.thrower_y IS NOT NULL
                    ORDER BY e2.event_number
                    LIMIT 1
                ) as next_thrower_y
            FROM events e1
            WHERE e1.event_type IN (20, 22)
                AND (
                    (e1.event_type = 20 AND e1.receiver_y IS NOT NULL)
                    OR (e1.event_type = 22 AND e1.turnover_y IS NOT NULL)
                )
        )
        SELECT
            game_id,
            turnover_y,
            next_thrower_y,
            turnover_y + next_thrower_y as sum_coords,
            ABS((turnover_y + next_thrower_y) - 120) as distance_from_120
        FROM turnovers
        WHERE next_thrower_y IS NOT NULL
        ORDER BY turnover_y
        LIMIT 30;
    """)

    results = cur.fetchall()

    print("\nSample of turnover coordinates and next possession starts:\n")
    print(f"{'Turnover Y':<12} {'Next Y':<10} {'Sum':<10} {'Dist from 120'}")
    print("-" * 50)

    sum_near_120 = 0
    total = 0

    for row in results:
        turnover_y = row['turnover_y']
        next_y = row['next_thrower_y']
        sum_coords = row['sum_coords']
        dist = row['distance_from_120']

        print(f"{turnover_y:<12.2f} {next_y:<10.2f} {sum_coords:<10.2f} {dist:.2f}")

        # Check if sum is close to 120 (within 10 yards)
        if dist < 10:
            sum_near_120 += 1
        total += 1

    print("\n" + "=" * 70)
    print(f"Turnovers where Y1 + Y2 ≈ 120: {sum_near_120}/{total} ({100*sum_near_120/total:.1f}%)")
    print("=" * 70)

    if sum_near_120 / total > 0.7:  # If more than 70% are near 120
        print("\n⚠️  WARNING: Coordinate averaging is problematic!")
        print("   Teams are recording from opposite orientations.")
        print("   Averaging them creates artificial midfield clustering.\n")
        print("   The issue from translationNotes.txt:")
        print("   - Team A records turnover at Y=34.83")
        print("   - Team B records possession at Y=80.12")
        print("   - Sum: 34.83 + 80.12 ≈ 120 (opposite orientations!)")
        print("   - Average: 57.48 (always near midfield - WRONG!)\n")
        print("   Solution: Use ONE team's coordinates, not the average.")
    else:
        print("\n✅ Coordinates look OK - averaging seems appropriate.")

    # Check distribution of turnover Y coordinates
    cur.execute("""
        SELECT
            FLOOR((CASE
                WHEN event_type = 20 THEN receiver_y
                WHEN event_type = 22 THEN turnover_y
            END) / 10) * 10 as y_bucket,
            COUNT(*) as count
        FROM events
        WHERE event_type IN (20, 22)
            AND (
                (event_type = 20 AND receiver_y IS NOT NULL AND receiver_y BETWEEN 20 AND 100)
                OR (event_type = 22 AND turnover_y IS NOT NULL AND turnover_y BETWEEN 20 AND 100)
            )
        GROUP BY y_bucket
        ORDER BY y_bucket;
    """)

    print("\n" + "=" * 70)
    print("TURNOVER DISTRIBUTION BY FIELD POSITION (non-endzone)")
    print("=" * 70)
    print(f"\n{'Field Y Range':<15} {'Count':<10} {'Bar'}")
    print("-" * 50)

    buckets = cur.fetchall()
    max_count = max(b['count'] for b in buckets) if buckets else 1

    midfield_count = 0
    total_turnovers = 0

    for bucket in buckets:
        y = bucket['y_bucket']
        count = bucket['count']
        bar = '█' * int(40 * count / max_count)
        print(f"{int(y):3d}-{int(y+10):3d} yards  {count:<10} {bar}")

        # Check if this is in the midfield zone (40-70 yards)
        if 40 <= y < 70:
            midfield_count += count
        total_turnovers += count

    print(f"\n{'='*50}")
    print(f"Turnovers in midfield (40-70 yards): {midfield_count}/{total_turnovers} ({100*midfield_count/total_turnovers:.1f}%)")
    print(f"{'='*50}")

    if midfield_count / total_turnovers > 0.5:
        print("\n⚠️  Over 50% of turnovers are clustered in midfield!")
        print("   This is likely due to the averaging issue.\n")

    cur.close()
    conn.close()

if __name__ == "__main__":
    check_averaging_issue()
